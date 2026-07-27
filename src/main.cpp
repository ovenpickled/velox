#include <iostream>
#include <string>
#include <vector>
#include <chrono>
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

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc) model_dir = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--batch-size" && i+1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--temperature" && i+1 < argc) { temperature = std::stof(argv[++i]); greedy = false; }
        else if (arg == "--greedy") greedy = true;
    }

    if (model_dir.empty()) {
        std::cerr << "Usage: inference --model <path> [--prompt <text>] [--max-tokens N] "
                  << "[--batch-size N] [--temperature T] [--greedy]" << std::endl;
        return 1;
    }

    std::cout << "Loading model from " << model_dir << "..." << std::endl;
    Qwen2Model model;
    if (!model.load(model_dir, batch_size)) {
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

    if (batch_size == 1) {
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
