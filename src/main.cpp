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
    float temperature = 1.0f;
    bool greedy = true;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc) model_dir = argv[++i];
        else if (arg == "--prompt" && i+1 < argc) prompt = argv[++i];
        else if (arg == "--max-tokens" && i+1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--temperature" && i+1 < argc) { temperature = std::stof(argv[++i]); greedy = false; }
        else if (arg == "--greedy") greedy = true;
    }
    
    if (model_dir.empty()) {
        std::cerr << "Usage: inference --model <path> [--prompt <text>] [--max-tokens N] [--temperature T] [--greedy]" << std::endl;
        return 1;
    }
    
    std::cout << "Loading model from " << model_dir << "..." << std::endl;
    Qwen2Model model;
    if (!model.load(model_dir)) { 
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
    std::cout << "Encoded prompt into " << input_tokens.size() << " tokens: [";
    for (size_t i = 0; i < input_tokens.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << input_tokens[i];
    }
    std::cout << "]" << std::endl;
    std::cout << prompt << std::flush;
    
    std::vector<float> logits(model.config().vocab_size);
    
    auto start_time = high_resolution_clock::now();
    
    model.forward(input_tokens, logits.data());
    int next_token = sampler.sample(logits.data(), model.config().vocab_size);
    int generated_count = 1;
    
    std::cout << tokenizer.decode(next_token) << std::flush;
    
    auto first_token_time = high_resolution_clock::now();
    
    for (int i = 1; i < max_tokens; i++) {
        if (next_token == tokenizer.eos_token_id()) break;
        
        model.forward({next_token}, logits.data());
        next_token = sampler.sample(logits.data(), model.config().vocab_size);
        generated_count++;
        
        std::cout << tokenizer.decode(next_token) << std::flush;
    }
    
    auto end_time = high_resolution_clock::now();
    
    std::cout << std::endl;
    auto prefill_duration = duration_cast<milliseconds>(first_token_time - start_time).count();
    auto total_duration = duration_cast<milliseconds>(end_time - start_time).count();
    auto decode_duration = total_duration - prefill_duration;
    
    std::cout << "\nGeneration stats:" << std::endl;
    std::cout << "- Total tokens generated: " << generated_count << std::endl;
    std::cout << "- Time to first token: " << prefill_duration << " ms" << std::endl;
    std::cout << "- Total time: " << total_duration << " ms" << std::endl;
    
    if (generated_count > 1 && decode_duration > 0) {
        float tokens_per_sec = (generated_count - 1) / (decode_duration / 1000.0f);
        std::cout << "- Decode speed: " << tokens_per_sec << " tokens/sec" << std::endl;
    }
    
    return 0;
}
