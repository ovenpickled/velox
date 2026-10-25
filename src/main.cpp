#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "model.hpp"
#include "tokenizer.hpp"
#include "sampler.hpp"

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

static float parse_float(const std::string& flag, const char* text) {
    try {
        size_t used = 0;
        float v = std::stof(text, &used);
        if (used != std::strlen(text)) throw std::invalid_argument("trailing characters");
        return v;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid value for " + flag + ": '" + text + "'");
    }
}

// Removes and returns the longest prefix of `pending` that ends on a UTF-8 character boundary.
// A multi-byte character can be split across tokens; the incomplete tail stays in `pending`
// until the next token completes it.
static std::string take_complete_utf8(std::string& pending) {
    const size_t n = pending.size();
    size_t keep = 0;  // trailing bytes that belong to a not-yet-complete character
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        unsigned char c = static_cast<unsigned char>(pending[n - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead byte
        size_t need = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
        if (need > back) keep = back;      // lead byte whose sequence is cut off
        break;
    }
    std::string out = pending.substr(0, n - keep);
    pending.erase(0, n - keep);
    return out;
}

static int run(int argc, char* argv[]) {
    std::string model_dir;
    std::string prompt = "Hello";
    int max_tokens = 64;
    int batch_size = 1;
    float temperature = 1.0f;
    bool greedy = true;
    bool benchmark = false;
    bool tokenize_only = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--model") model_dir = value();
        else if (arg == "--prompt") prompt = value();
        else if (arg == "--max-tokens") max_tokens = parse_int(arg, value());
        else if (arg == "--batch-size") batch_size = parse_int(arg, value());
        else if (arg == "--temperature") { temperature = parse_float(arg, value()); greedy = false; }
        else if (arg == "--greedy") greedy = true;
        else if (arg == "--benchmark") benchmark = true;
        else if (arg == "--tokenize-only") tokenize_only = true;
        else throw std::invalid_argument("unknown argument: " + arg);
    }

    if (model_dir.empty()) {
        std::cerr << "Usage: inference --model <path> [--prompt <text>] [--max-tokens N] "
                  << "[--batch-size N] [--temperature T] [--greedy] [--benchmark] [--tokenize-only]" << std::endl;
        return 1;
    }
    if (max_tokens < 1) throw std::invalid_argument("--max-tokens must be at least 1");
    if (batch_size < 1) throw std::invalid_argument("--batch-size must be at least 1");
    if (!greedy && temperature <= 0.0f) {
        std::cerr << "Note: --temperature <= 0 means greedy decoding" << std::endl;
        greedy = true;
    }

    // The tokenizer is cheap to load, so do it first: bad prompts fail before the model is loaded
    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir + "/tokenizer.json")) {
        std::cerr << "Failed to load tokenizer from " << model_dir << "/tokenizer.json" << std::endl;
        return 1;
    }
    auto input_tokens = tokenizer.encode(prompt);

    if (tokenize_only) {  // prints the ids as a JSON list, e.g. [9707,11]; used to compare against Hugging Face
        std::cout << "[";
        for (size_t i = 0; i < input_tokens.size(); i++) std::cout << (i ? "," : "") << input_tokens[i];
        std::cout << "]" << std::endl;
        return 0;
    }
    if (input_tokens.empty()) {
        throw std::invalid_argument("the prompt encodes to zero tokens; please give a non-empty --prompt");
    }

    std::cout << "Loading model from " << model_dir << "..." << std::endl;
    Qwen2Model model;
    int max_bs = benchmark ? std::max(batch_size, 8) : batch_size;
    if (!model.load(model_dir, max_bs)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    // The prompt takes L KV-cache slots and every further generated token takes one more
    const int ctx = model.max_context();
    const int prompt_len = static_cast<int>(input_tokens.size());
    if (prompt_len >= ctx) {
        throw std::invalid_argument("the prompt is " + std::to_string(prompt_len) +
                                    " tokens but the context limit is " + std::to_string(ctx));
    }
    if (max_tokens > ctx - prompt_len + 1) {
        max_tokens = ctx - prompt_len + 1;
        std::cerr << "Note: --max-tokens reduced to " << max_tokens << " to fit the " << ctx << "-token context limit" << std::endl;
    }

    SamplerConfig sampler_config;
    sampler_config.greedy = greedy;
    sampler_config.temperature = temperature;
    Sampler sampler(sampler_config);

    int vocab_size = model.config().vocab_size;

    std::cout << "Encoded prompt into " << input_tokens.size() << " tokens: [";
    for (size_t i = 0; i < input_tokens.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << input_tokens[i];
    }
    std::cout << "]" << std::endl;

    if (benchmark) {
        std::cout << "Running benchmark sweep (Prompt length: " << input_tokens.size() << ", Max decode: " << max_tokens << ")" << std::endl;
        std::cout << "Warming up engine (batch 1)..." << std::endl;
        
        // Warmup
        std::vector<float> warmup_logits(vocab_size);
        model.forward(input_tokens, warmup_logits.data());
        int warmup_token = sampler.sample(warmup_logits.data(), vocab_size);
        for(int i=1; i<8; i++) {
            model.forward({warmup_token}, warmup_logits.data());
            warmup_token = sampler.sample(warmup_logits.data(), vocab_size);
        }
        model.reset();

        std::cout << "\n| Batch Size | TTFT (ms) | Per-Seq Speed (tok/s) | Aggregate Throughput (tok/s) |" << std::endl;
        std::cout << "|------------|-----------|-----------------------|------------------------------|" << std::endl;

        std::vector<int> batches = {1, 2, 4, 8};
        int num_reps = 15;
        for (int bs : batches) {
            // Warmup this specific batch size
            std::vector<std::vector<int>> warmup_prompts(bs, input_tokens);
            std::vector<float> warmup_logits(bs * vocab_size);
            model.forward_batch(warmup_prompts, warmup_logits.data());
            model.forward_batch(std::vector<std::vector<int>>(bs, {0}), warmup_logits.data());
            model.reset();

            std::vector<float> ttfts, seq_tpss, agg_tpss;
            for (int rep = 0; rep < num_reps; rep++) {
                std::vector<std::vector<int>> batch_prompts(bs, input_tokens);
                std::vector<float> batch_logits(bs * vocab_size);
                std::vector<int> next_tokens(bs);
                std::vector<bool> done(bs, false);
                int active_count = bs;

                auto start_time = high_resolution_clock::now();

                model.forward_batch(batch_prompts, batch_logits.data());
                for (int b = 0; b < bs; b++) {
                    next_tokens[b] = sampler.sample(batch_logits.data() + b * vocab_size, vocab_size);
                    if (tokenizer.is_eos(next_tokens[b])) {   // a sequence can end on its very first token
                        done[b] = true;
                        active_count--;
                    }
                }

                auto first_token_time = high_resolution_clock::now();

                int total_generated = bs;
                for (int step = 1; step < max_tokens && active_count > 0; step++) {
                    std::vector<std::vector<int>> decode_tokens(bs);
                    for (int b = 0; b < bs; b++) {
                        decode_tokens[b] = {done[b] ? 0 : next_tokens[b]};
                    }

                    model.forward_batch(decode_tokens, batch_logits.data());

                    for (int b = 0; b < bs; b++) {
                        if (done[b]) continue;
                        next_tokens[b] = sampler.sample(batch_logits.data() + b * vocab_size, vocab_size);
                        total_generated++;

                        if (tokenizer.is_eos(next_tokens[b])) {
                            done[b] = true;
                            active_count--;
                        }
                    }
                }

                auto end_time = high_resolution_clock::now();

                auto prefill_ms = duration_cast<milliseconds>(first_token_time - start_time).count();
                auto total_ms = duration_cast<milliseconds>(end_time - start_time).count();
                auto decode_ms = total_ms - prefill_ms;

                float agg_tps = 0.0f;
                float seq_tps = 0.0f;
                if (total_generated > bs && decode_ms > 0) {
                    agg_tps = (total_generated - bs) / (decode_ms / 1000.0f);
                    seq_tps = agg_tps / bs;
                }

                ttfts.push_back(static_cast<float>(prefill_ms));
                seq_tpss.push_back(seq_tps);
                agg_tpss.push_back(agg_tps);
                model.reset();
            }

            auto calc_stats = [](std::vector<float>& vals, float& median, float& stdev) {
                float sum = 0;
                for (float v : vals) sum += v;
                float mean = sum / vals.size();
                float sq_sum = 0;
                for (float v : vals) sq_sum += (v - mean) * (v - mean);
                stdev = std::sqrt(sq_sum / vals.size());
                
                std::sort(vals.begin(), vals.end());
                median = vals[vals.size() / 2];
            };

            float med_ttft, std_ttft, med_seq, std_seq, med_agg, std_agg;
            calc_stats(ttfts, med_ttft, std_ttft);
            calc_stats(seq_tpss, med_seq, std_seq);
            calc_stats(agg_tpss, med_agg, std_agg);

            char ttft_str[32], seq_str[32], agg_str[32];
            snprintf(ttft_str, sizeof(ttft_str), "%.0f±%.0f", med_ttft, std_ttft);
            snprintf(seq_str, sizeof(seq_str), "%.2f±%.2f", med_seq, std_seq);
            snprintf(agg_str, sizeof(agg_str), "%.2f±%.2f", med_agg, std_agg);

            printf("| %-10d | %-9s | %-21s | %-28s |\n", bs, ttft_str, seq_str, agg_str);
        }

    } else if (batch_size == 1) {
        // --- Single Sequence Mode ---
        std::cout << prompt << std::flush;

        std::vector<float> logits(vocab_size);

        auto start_time = high_resolution_clock::now();

        model.forward(input_tokens, logits.data());
        if (const char* dump = std::getenv("VELOX_DUMP_LOGITS")) {
            std::ofstream f(dump, std::ios::binary);
            int32_t n = vocab_size;
            f.write(reinterpret_cast<const char*>(&n), sizeof(n));
            f.write(reinterpret_cast<const char*>(logits.data()), n * sizeof(float));
        }
        int next_token = sampler.sample(logits.data(), vocab_size);
        int generated_count = 1;

        // Print tokens as they arrive, holding back the bytes of a UTF-8 character split across tokens
        std::string pending;
        auto emit = [&](int token) {
            if (tokenizer.is_eos(token)) return;   // don't print <|endoftext|> and friends
            pending += tokenizer.decode(token);
            std::cout << take_complete_utf8(pending) << std::flush;
        };
        emit(next_token);

        auto first_token_time = high_resolution_clock::now();

        for (int i = 1; i < max_tokens; i++) {
            if (tokenizer.is_eos(next_token)) break;

            model.forward({next_token}, logits.data());
            next_token = sampler.sample(logits.data(), vocab_size);
            generated_count++;

            emit(next_token);
        }

        auto end_time = high_resolution_clock::now();

        std::cout << pending << std::endl;
        auto prefill_ms = duration_cast<milliseconds>(first_token_time - start_time).count();
        auto total_ms = duration_cast<milliseconds>(end_time - start_time).count();
        auto decode_ms = total_ms - prefill_ms;

        std::cout << "\nGeneration stats:" << std::endl;
        std::cout << "- Total tokens generated: " << generated_count << std::endl;
        std::cout << "- Time to first token: " << prefill_ms << " ms" << std::endl;
        std::cout << "- Total time: " << total_ms << " ms" << std::endl;

        if (generated_count > 1 && decode_ms > 0) {
            float tps = (generated_count - 1) / (decode_ms / 1000.0f);
            std::cout << "- Decode speed: " << tps << " tokens/sec" << std::endl;
        }

    } else {
        // --- Batch Mode ---
        std::cout << "Batch size: " << batch_size << std::endl;

        // Replicate prompt across batch
        std::vector<std::vector<int>> batch_prompts(batch_size, input_tokens);
        std::vector<float> batch_logits(batch_size * vocab_size);
        std::vector<int> next_tokens(batch_size);
        std::vector<bool> done(batch_size, false);
        std::vector<std::string> outputs(batch_size);
        int active_count = batch_size;

        auto start_time = high_resolution_clock::now();

        // Prefill
        model.forward_batch(batch_prompts, batch_logits.data());
        for (int b = 0; b < batch_size; b++) {
            next_tokens[b] = sampler.sample(batch_logits.data() + b * vocab_size, vocab_size);
            if (tokenizer.is_eos(next_tokens[b])) {   // a sequence can end on its very first token
                done[b] = true;
                active_count--;
            } else {
                outputs[b] += tokenizer.decode(next_tokens[b]);
            }
        }

        auto first_token_time = high_resolution_clock::now();

        // Decode
        int total_generated = batch_size;

        for (int step = 1; step < max_tokens && active_count > 0; step++) {
            std::vector<std::vector<int>> decode_tokens(batch_size);
            for (int b = 0; b < batch_size; b++) {
                decode_tokens[b] = {done[b] ? 0 : next_tokens[b]};
            }

            model.forward_batch(decode_tokens, batch_logits.data());

            for (int b = 0; b < batch_size; b++) {
                if (done[b]) continue;

                next_tokens[b] = sampler.sample(batch_logits.data() + b * vocab_size, vocab_size);
                total_generated++;

                if (tokenizer.is_eos(next_tokens[b])) {
                    done[b] = true;
                    active_count--;
                } else {
                    outputs[b] += tokenizer.decode(next_tokens[b]);
                }
            }
        }

        auto end_time = high_resolution_clock::now();

        // Display outputs
        for (int b = 0; b < batch_size; b++) {
            std::cout << "\n[Sequence " << b << "] " << prompt << outputs[b] << std::endl;
        }

        auto prefill_ms = duration_cast<milliseconds>(first_token_time - start_time).count();
        auto total_ms = duration_cast<milliseconds>(end_time - start_time).count();
        auto decode_ms = total_ms - prefill_ms;

        std::cout << "\nBatch generation stats:" << std::endl;
        std::cout << "- Batch size: " << batch_size << std::endl;
        std::cout << "- Total tokens generated: " << total_generated << std::endl;
        std::cout << "- Time to first token: " << prefill_ms << " ms" << std::endl;
        std::cout << "- Total time: " << total_ms << " ms" << std::endl;

        if (total_generated > batch_size && decode_ms > 0) {
            float aggregate_tps = (total_generated - batch_size) / (decode_ms / 1000.0f);
            float per_seq_tps = aggregate_tps / batch_size;
            std::cout << "- Aggregate throughput: " << aggregate_tps << " tokens/sec" << std::endl;
            std::cout << "- Per-sequence speed: " << per_seq_tps << " tokens/sec" << std::endl;
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
