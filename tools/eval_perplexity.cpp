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
#include <stdexcept>
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
    size_t target_bytes = 0;      // UTF-8 bytes of the scored target tokens
    double bits_per_byte = 0.0;   // tokenizer-independent quality metric
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

static EvalResult evaluate_sequence(Qwen2Model& model, const Tokenizer& tok, const std::vector<int>& all_tokens,
                                    size_t offset, int max_tokens, bool verbose = true) {
    EvalResult res;
    // Window = all_tokens[offset, offset + max_tokens]: max_tokens scored targets + 1 leading context token
    std::vector<int> tokens(all_tokens.begin() + offset,
                            all_tokens.begin() + std::min(all_tokens.size(), offset + static_cast<size_t>(max_tokens) + 1));
    int n = static_cast<int>(tokens.size());
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
        res.target_bytes += tok.decode(target_token).size();

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
    res.bits_per_byte = (res.target_bytes > 0) ? total_nll / std::log(2.0) / res.target_bytes : 0.0;
    res.tokens_per_sec = (res.elapsed_seconds > 0.0) ? (scored_tokens / res.elapsed_seconds) : 0.0;
    res.kv_memory_bytes = model.kv_cache_memory_bytes();
    res.kv_active_bytes = model.kv_cache_active_bytes();

    // KV traffic calculation:
    // Average KV cache length during generation is ~ n / 2
    // Bytes per token per layer: FP32 = 1024 B, INT8 = 280 B (K: one scale per 32 values, V: one per head)
    const ModelConfig& cfg = model.config();
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    size_t bytes_per_pos_per_layer = (model.kv_type() == KVType::FP32)
        ? (kv_dim * sizeof(float) * 2)
        : (kv_dim * sizeof(int8_t) * 2 + cfg.num_key_value_heads * sizeof(float) * (1 + ((cfg.head_dim % KVCache::KV_K_GROUP == 0) ? cfg.head_dim / KVCache::KV_K_GROUP : 1)));
    size_t bytes_per_pos_all_layers = bytes_per_pos_per_layer * cfg.num_hidden_layers;

    double avg_context = (scored_tokens) / 2.0;
    res.kv_traffic_mb_per_tok = (avg_context * bytes_per_pos_all_layers) / (1024.0 * 1024.0);
    res.kv_bandwidth_gb_s = (res.tokens_per_sec * res.kv_traffic_mb_per_tok) / 1024.0;

    return res;
}

// Long-context KV-cache benchmark at exact context depths (1024, 2048, 4096 tokens).
// Both models stay loaded; runs alternate FP32/INT8 for each repeat so thermal drift and
// background load hit both equally. Reports the median decode speed and the min-max range.
static void run_long_context_benchmark(const std::string& model_dir, const std::vector<int>& tokens) {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "TASK 3: INT8 KV-CACHE LONG-CONTEXT BENCHMARK (1K, 2K, 4K TOKENS)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    if (const char* e = std::getenv("VELOX_KV_INT8")) {
        if (std::string(e) == "1" || std::string(e) == "true")
            throw std::runtime_error("unset VELOX_KV_INT8: it would force the FP32 run to use an INT8 cache");
    }

    const std::vector<int> target_contexts = {1024, 2048, 4096};
    const int num_decode_steps = 128;   // timed decode steps at each depth
    const int repeats = 3;
    const int prefill_chunk = 512;

    Qwen2Model models[2];
    const KVType types[2] = {KVType::FP32, KVType::INT8};
    const char* names[2] = {"FP32", "INT8"};
    for (int i = 0; i < 2; i++) {
        if (!models[i].load(model_dir, 1, types[i]))
            throw std::runtime_error(std::string("failed to load model (") + names[i] + " KV run)");
    }
    std::vector<float> logits(models[0].config().vocab_size);

    // Prefill `ctx` tokens, then time `num_decode_steps` single-token decode steps. Returns tok/s.
    auto run_once = [&](Qwen2Model& model, int ctx) {
        model.reset();
        for (int s = 0; s < ctx; s += prefill_chunk) {
            int end_s = std::min(s + prefill_chunk, ctx);
            std::vector<int> chunk(tokens.begin() + s, tokens.begin() + end_s);
            model.forward(chunk, logits.data());
        }
        auto t0 = high_resolution_clock::now();
        for (int s = 0; s < num_decode_steps; s++) model.forward({tokens[ctx + s]}, logits.data());
        auto t1 = high_resolution_clock::now();
        return num_decode_steps / (duration_cast<microseconds>(t1 - t0).count() / 1e6);
    };

    struct Row { int ctx; double mem_mb[2]; double med[2], lo[2], hi[2]; };
    std::vector<Row> rows;

    for (int ctx : target_contexts) {
        if (static_cast<int>(tokens.size()) < ctx + num_decode_steps) {
            std::cout << "[SKIP] Not enough tokens for context " << ctx << std::endl;
            continue;
        }
        std::cout << ">>> Context depth " << ctx << ": " << repeats << " repeats x " << num_decode_steps << " decode steps" << std::endl;
        std::vector<double> speeds[2];
        for (int r = 0; r < repeats; r++) {
            for (int i = 0; i < 2; i++) {   // alternate FP32, INT8, FP32, INT8, ...
                double v = run_once(models[i], ctx);
                speeds[i].push_back(v);
                std::cout << "  repeat " << (r + 1) << "  " << names[i] << " KV: " << std::fixed << std::setprecision(2) << v << " tok/s" << std::endl;
            }
        }
        Row row{};
        row.ctx = ctx;
        for (int i = 0; i < 2; i++) {
            std::sort(speeds[i].begin(), speeds[i].end());
            row.med[i] = speeds[i][speeds[i].size() / 2];
            row.lo[i] = speeds[i].front();
            row.hi[i] = speeds[i].back();
            row.mem_mb[i] = models[i].kv_cache_active_bytes() / (1024.0 * 1024.0);   // analytic: from the cache layout
        }
        rows.push_back(row);
    }

    std::cout << "\n### Long-context KV-cache comparison (INT8 weights; median of " << repeats << " runs, range in brackets)\n" << std::endl;
    std::cout << "| Context | FP32 KV mem (MB) | INT8 KV mem (MB) | Memory saving | FP32 KV tok/s | INT8 KV tok/s | INT8 / FP32 speed |" << std::endl;
    std::cout << "|:-------:|-----------------:|-----------------:|--------------:|:-------------:|:-------------:|------------------:|" << std::endl;
    for (const auto& r : rows) {
        printf("| %7d | %16.2f | %16.2f | %12.1f%% | %5.2f [%5.2f-%5.2f] | %5.2f [%5.2f-%5.2f] | %16.2fx |\n",
               r.ctx, r.mem_mb[0], r.mem_mb[1], (1.0 - r.mem_mb[1] / r.mem_mb[0]) * 100.0,
               r.med[0], r.lo[0], r.hi[0], r.med[1], r.lo[1], r.hi[1], r.med[1] / r.med[0]);
    }
    std::cout << "Memory is computed from the cache layout (analytic), not measured. If the ranges overlap, the speed difference is within noise." << std::endl;
    std::cout << "For quality, use scripts/run_eval.py (4096 scored tokens); a 128-token PPL here would be too noisy to mean anything." << std::endl;
    std::cout << "================================================================================" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        std::string model_dir = "/home/awwwyan/models/Qwen2.5-0.5B";
        std::string file_path;
        int max_tokens = 1024;
        bool fp32 = false;
        bool bench_kv = false;
        int num_windows = 1;
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
            else if (arg == "--num-windows") num_windows = parse_int(arg, value());
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
        if (num_windows < 1) throw std::invalid_argument("--num-windows must be >= 1");
        EvalResult res;
        double sum_elapsed = 0.0;
        int windows_done = 0;
        for (int w = 0; w < num_windows; w++) {
            size_t offset = static_cast<size_t>(w) * max_tokens;   // consecutive, non-overlapping windows
            if (offset + 2 > tokens.size()) { std::cerr << "Only " << w << " full window(s) available in the input" << std::endl; break; }
            EvalResult r = evaluate_sequence(model, tokenizer, tokens, offset, max_tokens, true);
            if (r.num_tokens == 0) break;
            res.num_tokens += r.num_tokens;
            res.total_nll += r.total_nll;
            res.target_bytes += r.target_bytes;
            sum_elapsed += r.elapsed_seconds;
            res.kv_memory_bytes = std::max(res.kv_memory_bytes, r.kv_memory_bytes);
            res.kv_active_bytes = std::max(res.kv_active_bytes, r.kv_active_bytes);
            res.kv_traffic_mb_per_tok = r.kv_traffic_mb_per_tok;
            res.kv_bandwidth_gb_s = r.kv_bandwidth_gb_s;
            windows_done++;
        }
        if (res.num_tokens == 0) { std::cerr << "Nothing was evaluated" << std::endl; return 1; }
        res.mean_loss = res.total_nll / res.num_tokens;
        res.perplexity = std::exp(res.mean_loss);
        res.bits_per_byte = res.total_nll / std::log(2.0) / res.target_bytes;
        res.elapsed_seconds = sum_elapsed;
        res.tokens_per_sec = res.num_tokens / sum_elapsed;

        std::cout << "\n================================================================================" << std::endl;
        std::cout << "EVALUATION RESULTS:" << std::endl;
        std::cout << "  Windows:             " << windows_done << " x " << max_tokens << " tokens" << std::endl;
        std::cout << "  Tokens Evaluated:    " << res.num_tokens << std::endl;
        std::cout << "  Cross-Entropy Loss:  " << std::fixed << std::setprecision(4) << res.mean_loss << std::endl;
        std::cout << "  Perplexity (PPL):    " << std::fixed << std::setprecision(4) << res.perplexity << std::endl;
        std::cout << "  Bits per byte:       " << std::fixed << std::setprecision(4) << res.bits_per_byte << "  (comparable across languages)" << std::endl;
        std::cout << "  Evaluation Speed:    " << std::fixed << std::setprecision(2) << res.tokens_per_sec << " tokens/sec" << std::endl;
        std::cout << "  Elapsed Time:        " << std::fixed << std::setprecision(2) << res.elapsed_seconds << " seconds" << std::endl;
        std::cout << "  KV Cache Active Mem: " << std::fixed << std::setprecision(2) << res.kv_active_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Avg KV Cache Traffic:" << std::fixed << std::setprecision(2) << res.kv_traffic_mb_per_tok << " MB / token" << std::endl;
        std::cout << "  Avg KV Bandwidth:    " << std::fixed << std::setprecision(2) << res.kv_bandwidth_gb_s << " GB / sec" << std::endl;
        std::cout << "================================================================================" << std::endl;
        // One machine-readable line for scripts/run_eval.py
        std::cout << "RESULT_JSON {\"weights\":\"" << (fp32 ? "fp32" : "int8") << "\",\"kv\":\""
                  << (kv_type == KVType::INT8 ? "int8" : "fp32") << "\",\"tokens\":" << res.num_tokens
                  << ",\"ppl\":" << std::setprecision(6) << res.perplexity << ",\"loss\":" << res.mean_loss
                  << ",\"bpb\":" << res.bits_per_byte << ",\"tok_s\":" << res.tokens_per_sec << "}" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}