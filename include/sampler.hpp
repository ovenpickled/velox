#pragma once
#include <random>

struct SamplerConfig {
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    bool greedy = true;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig config = {});

    int sample(const float* logits, int vocab_size);

private:
    SamplerConfig config_;
    std::mt19937 rng_;

    int greedy_sample(const float* logits, int vocab_size);
    int top_k_top_p_sample(const float* logits, int vocab_size);
};
