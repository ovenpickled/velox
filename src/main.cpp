#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "model.hpp"
#include "tokenizer.hpp"
#include "sampler.hpp"

using namespace std::chrono;

int main(int argc, char* argv[]) {
    std::string model_dir;
    std::string prompt = "Hello";
    int max_tokens = 64;
    int batch_size = 1;
    float temperature = 1.0f;
    bool greedy = true;
    bool benchmark = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc) model_dir = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--batch-size" && i+1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--temperature" && i+1 < argc) { temperature = std::stof(argv[++i]); greedy = false; }
        else if (arg == "--greedy") greedy = true;
        else if (arg == "--benchmark") benchmark = true;
    }

    if (model_dir.empty()) {
        std::cerr << "Usage: inference --model <path> [--prompt <text>] [--max-tokens N] "
                  << "[--batch-size N] [--temperature T] [--greedy]" << std::endl;
        return 1;
    }

    std::cout << "Loading model from " << model_dir << "..." << std::endl;
    Qwen2Model model;
    int max_bs = benchmark ? std::max(batch_size, 8) : batch_size;
    if (!model.load(model_dir, max_bs)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    std::cout << "Loading tokenizer..." << std::endl;
    Tokenizer tokenizer;
    if (!tokenizer.load(model_dir + "/tokenizer.json")) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }

    SamplerConfig sampler_config;
    sampler_config.greedy = greedy;
    sampler_config.temperature = temperature;
    Sampler sampler(sampler_config);

    auto input_tokens = tokenizer.encode(prompt);
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

                        if (next_tokens[b] == tokenizer.eos_token_id()) {
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
        int next_token = sampler.sample(logits.data(), vocab_size);
        int generated_count = 1;

        std::cout << tokenizer.decode(next_token) << std::flush;

        auto first_token_time = high_resolution_clock::now();

        for (int i = 1; i < max_tokens; i++) {
            if (next_token == tokenizer.eos_token_id()) break;

            model.forward({next_token}, logits.data());
            next_token = sampler.sample(logits.data(), vocab_size);
            generated_count++;

            std::cout << tokenizer.decode(next_token) << std::flush;
        }

        auto end_time = high_resolution_clock::now();

        std::cout << std::endl;
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
            outputs[b] += tokenizer.decode(next_tokens[b]);
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
                outputs[b] += tokenizer.decode(next_tokens[b]);
                total_generated++;

                if (next_tokens[b] == tokenizer.eos_token_id()) {
                    done[b] = true;
                    active_count--;
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
