#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "model.hpp"
#include "tokenizer.hpp"

using namespace std::chrono;

static int parse_int(const std::string& flag, const char* text) {
    try {
        size_t used = 0;
        int v = std::stoi(text, &used);
        if (used != std::strlen(text)) throw std::invalid_argument("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + flag + ": '" + text + "'");
    }
}

struct EvalResult {
    int num_tokens = 0;
    double total_nll = 0.0;
    double mean_loss = 0.0;
    double perplexity = 0.0;
    double elapsed_seconds = 0.0;
    double tokens_per_sec = 0.0;
    size_t kv_memory_bytes = 0;
    size_t kv_active_bytes = 0;
    double kv_traffic_mb_per_tok = 0.0;
    double kv_bandwidth_gb_s = 0.0;
};

// Computes Negative Log Likelihood (NLL) of target_token given logits
static double compute_nll(const float* logits, int vocab_size, int target_token) {
    float max_l = logits[0];
    for (int v = 1; v < vocab_size; v++) {
        if (logits[v] > max_l) max_l = logits[v];
    }
    double sum_exp = 0.0;
    #pragma omp parallel for reduction(+:sum_exp) schedule(static)
    for (int v = 0; v < vocab_size; v++) {
        sum_exp += std::exp(logits[v] - max_l);
    }
    double lse = max_l + std::log(sum_exp);
    return lse - static_cast<double>(logits[target_token]);
}

static EvalResult evaluate_sequence(Qwen2Model& model, const std::vector<int>& tokens, int max_tokens, bool verbose = true) {
    EvalResult res;
    int n = std::min(static_cast<int>(tokens.size()), max_tokens + 1);
    if (n < 2) {
        std::cerr << "Not enough tokens to evaluate (need at least 2 tokens, got " << n << ")" << std::endl;
        return res;
    }

    int vocab_size = model.config().vocab_size;
    std::vector<float> logits(vocab_size);

    model.reset();
    auto start_time = high_resolution_clock::now();

    int scored_tokens = 0;
    double total_nll = 0.0;

    // Autoregressively feed tokens one by one into the KV cache and score the next token
    for (int t = 0; t < n - 1; t++) {
        int current_token = tokens[t];
        int target_token = tokens[t + 1];

        model.forward({current_token}, logits.data());

        double nll = compute_nll(logits.data(), vocab_size, target_token);
        total_nll += nll;
        scored_tokens++;

        if (verbose && ((scored_tokens % 64 == 0) || (scored_tokens == n - 1))) {
            double cur_loss = total_nll / scored_tokens;
            double cur_ppl = std::exp(cur_loss);
            auto cur_time = high_resolution_clock::now();
            double cur_sec = duration_cast<milliseconds>(cur_time - start_time).count() / 1000.0;
            double cur_tps = (cur_sec > 0.0) ? (scored_tokens / cur_sec) : 0.0;
            double kv_mb = model.kv_cache_active_bytes() / (1024.0 * 1024.0);

            std::cout << "\r[" << std::setw(5) << scored_tokens << "/" << std::setw(5) << (n - 1) << "] "
                      << "Loss: " << std::fixed << std::setprecision(4) << cur_loss
                      << " | PPL: " << std::setw(8) << std::setprecision(3) << cur_ppl
                      << " | Speed: " << std::setw(5) << std::setprecision(1) << cur_tps << " tok/s"
                      << " | KV Mem: " << std::setw(5) << std::setprecision(2) << kv_mb << " MB"
                      << std::flush;
        }
    }
    if (verbose) std::cout << std::endl;

    auto end_time = high_resolution_clock::now();
    res.elapsed_seconds = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;
    res.num_tokens = scored_tokens;
    res.total_nll = total_nll;
    res.mean_loss = total_nll / scored_tokens;
    res.perplexity = std::exp(res.mean_loss);
    res.tokens_per_sec = (res.elapsed_seconds > 0.0) ? (scored_tokens / res.elapsed_seconds) : 0.0;
    res.kv_memory_bytes = model.kv_cache_memory_bytes();
    res.kv_active_bytes = model.kv_cache_active_bytes();

    // KV traffic calculation:
    // Average KV cache length during generation is ~ n / 2
    // Bytes per token per layer: FP32 = 1024 B, INT8 = 272 B
    const ModelConfig& cfg = model.config();
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    size_t bytes_per_pos_per_layer = (model.kv_type() == KVType::FP32)
        ? (kv_dim * sizeof(float) * 2)
        : (kv_dim * sizeof(int8_t) * 2 + cfg.num_key_value_heads * sizeof(float) * 2);
    size_t bytes_per_pos_all_layers = bytes_per_pos_per_layer * cfg.num_hidden_layers;

    double avg_context = (scored_tokens) / 2.0;
    res.kv_traffic_mb_per_tok = (avg_context * bytes_per_pos_all_layers) / (1024.0 * 1024.0);
    res.kv_bandwidth_gb_s = (res.tokens_per_sec * res.kv_traffic_mb_per_tok) / 1024.0;

    return res;
}

// Long-context benchmark measuring at exact target context depths (1024, 2048, 4096 tokens)
static void run_long_context_benchmark(const std::string& model_dir, const std::vector<int>& tokens) {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "TASK 3: INT8 KV-CACHE LONG-CONTEXT BENCHMARK (1K, 2K, 4K TOKENS)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::vector<int> target_contexts = {1024, 2048, 4096};
    const int num_decode_steps = 32; // decode steps measured at context depth

    std::cout << "Benchmarking KV traffic, memory, and decode latency at context depths: 1k, 2k, 4k...\n" << std::endl;

    struct ContextResult {
        int context = 0;
        double fp32_speed = 0.0;
        double int8_speed = 0.0;
        double fp32_mem_mb = 0.0;
        double int8_mem_mb = 0.0;
        double fp32_traffic_mb = 0.0;
        double int8_traffic_mb = 0.0;
        double fp32_ppl = 0.0;
        double int8_ppl = 0.0;
    };
    std::vector<ContextResult> results;

    for (int ctx : target_contexts) {
        if (static_cast<int>(tokens.size()) < ctx + num_decode_steps) {
            std::cout << "[SKIP] Not enough tokens for context " << ctx << std::endl;
            continue;
        }

        ContextResult cr;
        cr.context = ctx;

        std::cout << ">>> Context depth: " << ctx << " tokens (" << num_decode_steps << " decode steps)" << std::endl;

        // Run FP32 KV Cache
        {
            std::cout << "  [Running FP32 KV Cache]..." << std::flush;
            Qwen2Model model_fp32;
            model_fp32.load(model_dir, 1, KVType::FP32);
            std::vector<float> logits(model_fp32.config().vocab_size);

            // Chunked prefill up to ctx (chunks of 512 for cache locality)
            const int prefill_chunk = 512;
            for (int s = 0; s < ctx; s += prefill_chunk) {
                int end_s = std::min(s + prefill_chunk, ctx);
                std::vector<int> chunk(tokens.begin() + s, tokens.begin() + end_s);
                model_fp32.forward(chunk, logits.data());
            }

            // Measure decode of next num_decode_steps tokens
            double nll_sum = 0.0;
            auto t0 = high_resolution_clock::now();
            for (int s = 0; s < num_decode_steps; s++) {
                int tok = tokens[ctx + s];
                int target = tokens[ctx + s + 1];
                model_fp32.forward({tok}, logits.data());
                nll_sum += compute_nll(logits.data(), model_fp32.config().vocab_size, target);
            }
            auto t1 = high_resolution_clock::now();
            double secs = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;

            cr.fp32_speed = num_decode_steps / secs;
            cr.fp32_mem_mb = model_fp32.kv_cache_active_bytes() / (1024.0 * 1024.0);
            cr.fp32_traffic_mb = (ctx * 24 * 1024.0) / (1024.0 * 1024.0); // 24 KB per token
            cr.fp32_ppl = std::exp(nll_sum / num_decode_steps);
            std::cout << " Speed: " << cr.fp32_speed << " tok/s | Memory: " << cr.fp32_mem_mb << " MB | PPL: " << cr.fp32_ppl << std::endl;
        }

        // Run INT8 KV Cache
        {
            std::cout << "  [Running INT8 KV Cache]..." << std::flush;
            Qwen2Model model_int8;
            model_int8.load(model_dir, 1, KVType::INT8);
            std::vector<float> logits(model_int8.config().vocab_size);

            // Chunked prefill up to ctx (chunks of 512 for cache locality)
            const int prefill_chunk = 512;
            for (int s = 0; s < ctx; s += prefill_chunk) {
                int end_s = std::min(s + prefill_chunk, ctx);
                std::vector<int> chunk(tokens.begin() + s, tokens.begin() + end_s);
                model_int8.forward(chunk, logits.data());
            }

            // Measure decode of next num_decode_steps tokens
            double nll_sum = 0.0;
            auto t0 = high_resolution_clock::now();
            for (int s = 0; s < num_decode_steps; s++) {
                int tok = tokens[ctx + s];
                int target = tokens[ctx + s + 1];
                model_int8.forward({tok}, logits.data());
                nll_sum += compute_nll(logits.data(), model_int8.config().vocab_size, target);
            }
            auto t1 = high_resolution_clock::now();
            double secs = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;

            cr.int8_speed = num_decode_steps / secs;
            cr.int8_mem_mb = model_int8.kv_cache_active_bytes() / (1024.0 * 1024.0);
            cr.int8_traffic_mb = (ctx * 24 * 272.0) / (1024.0 * 1024.0); // 6.375 KB per token
            cr.int8_ppl = std::exp(nll_sum / num_decode_steps);
            std::cout << " Speed: " << cr.int8_speed << " tok/s | Memory: " << cr.int8_mem_mb << " MB | PPL: " << cr.int8_ppl << std::endl;
        }

        results.push_back(cr);
    }

    // Print benchmark comparison table
    std::cout << "\n### Long Context (1k, 2k, 4k) KV-Cache Comparison Table\n" << std::endl;
    std::cout << "| Context Depth | FP32 Mem (MB) | INT8 Mem (MB) | Memory Saving | FP32 Traffic (MB/tok) | INT8 Traffic (MB/tok) | Traffic Reduction | FP32 Speed (tok/s) | INT8 Speed (tok/s) | Speedup | PPL Delta |" << std::endl;
    std::cout << "|:-------------:|:-------------:|:-------------:|:-------------:|:---------------------:|:---------------------:|:-----------------:|:------------------:|:------------------:|:-------:|:---------:|" << std::endl;

    for (const auto& r : results) {
        double mem_saving = (1.0 - r.int8_mem_mb / r.fp32_mem_mb) * 100.0;
        double traffic_reduction = r.fp32_traffic_mb / r.int8_traffic_mb;
        double speedup = r.int8_speed / r.fp32_speed;
        double ppl_diff = ((r.int8_ppl - r.fp32_ppl) / r.fp32_ppl) * 100.0;

        printf("| %13d | %13.2f | %13.2f | %12.1f%% | %21.2f | %21.2f | %16.2fx | %18.2f | %18.2f | %6.2fx | %+8.3f%% |\n",
               r.context, r.fp32_mem_mb, r.int8_mem_mb, mem_saving,
               r.fp32_traffic_mb, r.int8_traffic_mb, traffic_reduction,
               r.fp32_speed, r.int8_speed, speedup, ppl_diff);
    }
    std::cout << "================================================================================" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        std::string model_dir = "/home/awwwyan/models/Qwen2.5-0.5B";
        std::string file_path;
        int max_tokens = 1024;
        bool fp32 = false;
        bool bench_kv = false;
        KVType kv_type = KVType::FP32;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            auto value = [&]() -> const char* {
                if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--model") model_dir = value();
            else if (arg == "--file") file_path = value();
            else if (arg == "--max-tokens") max_tokens = parse_int(arg, value());
            else if (arg == "--fp32") fp32 = true;
            else if (arg == "--benchmark-kv") bench_kv = true;
            else if (arg == "--kv-cache-type") {
                std::string t = value();
                if (t == "int8" || t == "INT8" || t == "q8_0") kv_type = KVType::INT8;
                else if (t == "fp32" || t == "FP32" || t == "f32") kv_type = KVType::FP32;
                else throw std::invalid_argument("unknown --kv-cache-type: " + t);
            }
            else throw std::invalid_argument("unknown argument: " + arg);
        }

        if (fp32) {
            setenv("VELOX_FP32_LAYERS", "all", 1);
            setenv("VELOX_FP32_LMHEAD", "1", 1);
        }

        std::cout << "================================================================================" << std::endl;
        std::cout << "VELOX PERPLEXITY EVALUATION HARNESS" << std::endl;
        std::cout << "Model:         " << model_dir << std::endl;
        std::cout << "Weights Mode:  " << (fp32 ? "FP32 (Unquantized)" : "INT8 (W8A8 Group-128)") << std::endl;
        std::cout << "KV Cache Mode: " << (kv_type == KVType::INT8 ? "INT8 Symmetric (Per-Token-Head)" : "FP32") << std::endl;
        std::cout << "Max Tokens:    " << max_tokens << std::endl;
        std::cout << "Input File:    " << (file_path.empty() ? "(none)" : file_path) << std::endl;
        std::cout << "================================================================================" << std::endl;

        // Load tokenizer
        Tokenizer tokenizer;
        if (!tokenizer.load(model_dir + "/tokenizer.json")) {
            std::cerr << "Failed to load tokenizer from " << model_dir << "/tokenizer.json" << std::endl;
            return 1;
        }

        // Read input file
        std::string text;
        if (!file_path.empty()) {
            std::ifstream in(file_path);
            if (!in.is_open()) {
                std::cerr << "Could not open file: " << file_path << std::endl;
                return 1;
            }
            std::stringstream ss;
            ss << in.rdbuf();
            text = ss.str();
        } else {
            text = "The quick brown fox jumps over the lazy dog. Artificial intelligence and deep learning have transformed language modeling.";
        }

        std::cout << "Tokenizing input text (" << text.size() << " bytes)..." << std::flush;
        auto tokens = tokenizer.encode(text);
        std::cout << " " << tokens.size() << " tokens generated." << std::endl;

        if (bench_kv) {
            run_long_context_benchmark(model_dir, tokens);
            return 0;
        }

        // Load model
        std::cout << "Loading model weights..." << std::endl;
        Qwen2Model model;
        if (!model.load(model_dir, 1, kv_type)) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }

        std::cout << "Starting evaluation...\n" << std::endl;
        EvalResult res = evaluate_sequence(model, tokens, max_tokens, true);

        std::cout << "\n================================================================================" << std::endl;
        std::cout << "EVALUATION RESULTS:" << std::endl;
        std::cout << "  Tokens Evaluated:    " << res.num_tokens << std::endl;
        std::cout << "  Cross-Entropy Loss:  " << std::fixed << std::setprecision(4) << res.mean_loss << std::endl;
        std::cout << "  Perplexity (PPL):    " << std::fixed << std::setprecision(4) << res.perplexity << std::endl;
        std::cout << "  Evaluation Speed:    " << std::fixed << std::setprecision(2) << res.tokens_per_sec << " tokens/sec" << std::endl;
        std::cout << "  Elapsed Time:        " << std::fixed << std::setprecision(2) << res.elapsed_seconds << " seconds" << std::endl;
        std::cout << "  KV Cache Active Mem: " << std::fixed << std::setprecision(2) << res.kv_active_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Avg KV Cache Traffic:" << std::fixed << std::setprecision(2) << res.kv_traffic_mb_per_tok << " MB / token" << std::endl;
        std::cout << "  Avg KV Bandwidth:    " << std::fixed << std::setprecision(2) << res.kv_bandwidth_gb_s << " GB / sec" << std::endl;
        std::cout << "================================================================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
