#include "sampler.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

Sampler::Sampler(SamplerConfig config) : config_(config) {
    std::random_device rd;
    rng_.seed(rd());
}

int Sampler::sample(const float* logits, int vocab_size) {
    if (config_.greedy) {
        return greedy_sample(logits, vocab_size);
    }
    return top_k_top_p_sample(logits, vocab_size);
}

int Sampler::greedy_sample(const float* logits, int vocab_size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int Sampler::top_k_top_p_sample(const float* logits, int vocab_size) {
    std::vector<std::pair<float, int>> probs;
    probs.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        probs.push_back({logits[i] / config_.temperature, i});
    }

    if (config_.top_k > 0 && config_.top_k < vocab_size) {
        std::partial_sort(probs.begin(), probs.begin() + config_.top_k, probs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        probs.resize(config_.top_k);
    } else {
        std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    float max_logit = probs[0].first;
    float sum_exp = 0.0f;
    for (auto& p : probs) {
        p.first = std::exp(p.first - max_logit);
        sum_exp += p.first;
    }
    for (auto& p : probs) {
        p.first /= sum_exp;
    }

    if (config_.top_p < 1.0f) {
        float cumulative_prob = 0.0f;
        int keep_count = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative_prob += probs[i].first;
            keep_count++;
            if (cumulative_prob >= config_.top_p) {
                break;
            }
        }
        probs.resize(keep_count);
    }

    std::vector<float> final_probs;
    final_probs.reserve(probs.size());
    for (const auto& p : probs) {
        final_probs.push_back(p.first);
    }

    std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
    int sampled_idx = dist(rng_);
    return probs[sampled_idx].second;
}
