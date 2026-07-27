#include "model.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include "json.hpp"
#include "ops.hpp"

using json = nlohmann::json;

// --- KVCache ---

void KVCache::init(int num_layers, int num_kv_heads, int head_dim, int max_seq) {
    this->max_seq_len = max_seq;
    this->kv_dim = num_kv_heads * head_dim;
    seq_len = 0;

    size_t layer_size = static_cast<size_t>(max_seq) * kv_dim;
    k_cache.resize(num_layers);
    v_cache.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        k_cache[i].resize(layer_size, 0.0f);
        v_cache[i].resize(layer_size, 0.0f);
    }
}

void KVCache::reset() {
    seq_len = 0;
}

// --- ModelConfig ---

ModelConfig ModelConfig::from_json(const std::string& config_json_path) {
    ModelConfig config;
    std::ifstream f(config_json_path);
    if (!f.is_open()) return config;
    
    json j;
    f >> j;

    if (j.contains("hidden_size")) config.hidden_size = j["hidden_size"];
    if (j.contains("num_hidden_layers")) config.num_hidden_layers = j["num_hidden_layers"];
    if (j.contains("num_attention_heads")) config.num_attention_heads = j["num_attention_heads"];
    if (j.contains("num_key_value_heads")) config.num_key_value_heads = j["num_key_value_heads"];
    if (j.contains("intermediate_size")) config.intermediate_size = j["intermediate_size"];
    if (j.contains("vocab_size")) config.vocab_size = j["vocab_size"];
    if (j.contains("max_position_embeddings")) config.max_position_embeddings = j["max_position_embeddings"];
    if (j.contains("rope_theta")) config.rope_theta = j["rope_theta"];
    if (j.contains("rms_norm_eps")) config.rms_norm_eps = j["rms_norm_eps"];
    if (j.contains("tie_word_embeddings")) config.tie_word_embeddings = j["tie_word_embeddings"];
    
    config.head_dim = config.hidden_size / config.num_attention_heads;
    return config;
}

// --- Qwen2Model ---

const ModelConfig& Qwen2Model::config() const {
    return config_;
}

bool Qwen2Model::load(const std::string& model_dir) {
    config_ = ModelConfig::from_json(model_dir + "/config.json");
    
    if (!safetensors_.open(model_dir + "/model.safetensors")) {
        std::cerr << "Failed to open safetensors file: " << model_dir << "/model.safetensors" << std::endl;
        return false;
    }
    
    load_weights();

    int cache_max = std::min(config_.max_position_embeddings, 4096);
    kv_cache_.init(config_.num_hidden_layers, config_.num_key_value_heads, config_.head_dim, cache_max);

    allocate_buffers(1, 1);
    return true;
}

void Qwen2Model::reset() {
    kv_cache_.reset();
}

void Qwen2Model::load_weights() {
    weights_.layers.resize(config_.num_hidden_layers);
    
    auto get_tensor_data = [&](const std::string& name) -> const float* {
        if (!safetensors_.has_tensor(name)) {
            return nullptr;
        }
        Tensor t = safetensors_.get_tensor(name);
        stored_tensors_.push_back(t);
        return stored_tensors_.back().data();
    };
    
    weights_.embed_tokens = get_tensor_data("model.embed_tokens.weight");
    weights_.final_norm = get_tensor_data("model.norm.weight");
    
    if (config_.tie_word_embeddings) {
        weights_.lm_head = weights_.embed_tokens;
    } else {
        weights_.lm_head = get_tensor_data("lm_head.weight");
    }
    
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string prefix = "model.layers." + std::to_string(i);
        LayerWeights& lw = weights_.layers[i];
        
        lw.q_proj_w = get_tensor_data(prefix + ".self_attn.q_proj.weight");
        lw.q_proj_b = get_tensor_data(prefix + ".self_attn.q_proj.bias");
        lw.k_proj_w = get_tensor_data(prefix + ".self_attn.k_proj.weight");
        lw.k_proj_b = get_tensor_data(prefix + ".self_attn.k_proj.bias");
        lw.v_proj_w = get_tensor_data(prefix + ".self_attn.v_proj.weight");
        lw.v_proj_b = get_tensor_data(prefix + ".self_attn.v_proj.bias");
        lw.o_proj_w = get_tensor_data(prefix + ".self_attn.o_proj.weight");
        
        lw.gate_proj_w = get_tensor_data(prefix + ".mlp.gate_proj.weight");
        lw.up_proj_w = get_tensor_data(prefix + ".mlp.up_proj.weight");
        lw.down_proj_w = get_tensor_data(prefix + ".mlp.down_proj.weight");
        
        lw.input_layernorm_w = get_tensor_data(prefix + ".input_layernorm.weight");
        lw.post_attn_layernorm_w = get_tensor_data(prefix + ".post_attention_layernorm.weight");
    }
}

void Qwen2Model::allocate_buffers(int num_new, int total_len) {
    int hidden_size = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim;
    int intermediate_size = config_.intermediate_size;

    hidden_.resize(num_new * hidden_size);
    residual_.resize(num_new * hidden_size);
    norm_out_.resize(std::max(num_new * hidden_size, hidden_size));
    q_buf_.resize(num_new * num_heads * head_dim);
    k_buf_.resize(num_new * num_kv_heads * head_dim);
    v_buf_.resize(num_new * num_kv_heads * head_dim);
    attn_out_.resize(num_new * num_heads * head_dim);
    attn_scores_.resize(static_cast<size_t>(num_heads) * num_new * total_len);
    ffn_gate_.resize(num_new * intermediate_size);
    ffn_up_.resize(num_new * intermediate_size);
}

void Qwen2Model::forward(const std::vector<int>& tokens, float* logits) {
    int num_new = tokens.size();
    int past_len = kv_cache_.seq_len;
    int total_len = past_len + num_new;

    allocate_buffers(num_new, total_len);

    int hidden_size = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim;
    int intermediate_size = config_.intermediate_size;
    int kv_dim = num_kv_heads * head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int heads_per_group = num_heads / num_kv_heads;

    // --- Embedding ---
    for (int p = 0; p < num_new; p++) {
        ops::embedding_lookup(hidden_.data() + p * hidden_size, weights_.embed_tokens, tokens[p], hidden_size);
    }

    // --- Transformer Layers ---
    for (int layer = 0; layer < config_.num_hidden_layers; layer++) {
        LayerWeights& lw = weights_.layers[layer];
        float* k_cache = kv_cache_.k_cache[layer].data();
        float* v_cache = kv_cache_.v_cache[layer].data();

        // Residual
        std::memcpy(residual_.data(), hidden_.data(), num_new * hidden_size * sizeof(float));

        // Input LayerNorm
        for (int p = 0; p < num_new; p++) {
            ops::rmsnorm(norm_out_.data() + p * hidden_size,
                        hidden_.data() + p * hidden_size,
                        lw.input_layernorm_w, hidden_size, config_.rms_norm_eps);
        }

        // Q/K/V Projections
        ops::matmul(q_buf_.data(), norm_out_.data(), lw.q_proj_w, num_new, num_heads * head_dim, hidden_size);
        ops::matmul(k_buf_.data(), norm_out_.data(), lw.k_proj_w, num_new, kv_dim, hidden_size);
        ops::matmul(v_buf_.data(), norm_out_.data(), lw.v_proj_w, num_new, kv_dim, hidden_size);

        for (int p = 0; p < num_new; p++) {
            ops::add_bias(q_buf_.data() + p * num_heads * head_dim, lw.q_proj_b, num_heads * head_dim);
            ops::add_bias(k_buf_.data() + p * kv_dim, lw.k_proj_b, kv_dim);
            ops::add_bias(v_buf_.data() + p * kv_dim, lw.v_proj_b, kv_dim);
        }

        // RoPE (positions offset by past_len)
        for (int p = 0; p < num_new; p++) {
            ops::rope(q_buf_.data() + p * num_heads * head_dim,
                     k_buf_.data() + p * kv_dim,
                     head_dim, num_heads, num_kv_heads, past_len + p, config_.rope_theta);
        }

        // Append new K/V to cache
        std::memcpy(k_cache + past_len * kv_dim, k_buf_.data(), num_new * kv_dim * sizeof(float));
        std::memcpy(v_cache + past_len * kv_dim, v_buf_.data(), num_new * kv_dim * sizeof(float));

        // Attention: Q[num_new] × K_cache[total_len]
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_group;

            for (int qp = 0; qp < num_new; qp++) {
                int actual_qpos = past_len + qp;
                float* scores = attn_scores_.data() + (h * num_new + qp) * total_len;

                // Compute attention scores
                for (int kp = 0; kp < total_len; kp++) {
                    if (kp > actual_qpos) {
                        scores[kp] = -INFINITY;
                    } else {
                        float dot = 0;
                        for (int d = 0; d < head_dim; d++) {
                            dot += q_buf_[qp * num_heads * head_dim + h * head_dim + d]
                                 * k_cache[kp * kv_dim + kv_h * head_dim + d];
                        }
                        scores[kp] = dot * scale;
                    }
                }
                ops::softmax(scores, total_len);

                // Weighted sum of V
                for (int d = 0; d < head_dim; d++) {
                    float val = 0;
                    for (int kp = 0; kp < total_len; kp++) {
                        val += scores[kp] * v_cache[kp * kv_dim + kv_h * head_dim + d];
                    }
                    attn_out_[qp * num_heads * head_dim + h * head_dim + d] = val;
                }
            }
        }

        // Output Projection + Residual
        ops::matmul(hidden_.data(), attn_out_.data(), lw.o_proj_w, num_new, hidden_size, num_heads * head_dim);
        ops::add(hidden_.data(), hidden_.data(), residual_.data(), num_new * hidden_size);

        // FFN
        std::memcpy(residual_.data(), hidden_.data(), num_new * hidden_size * sizeof(float));

        for (int p = 0; p < num_new; p++) {
            ops::rmsnorm(norm_out_.data() + p * hidden_size,
                        hidden_.data() + p * hidden_size,
                        lw.post_attn_layernorm_w, hidden_size, config_.rms_norm_eps);
        }

        ops::matmul(ffn_gate_.data(), norm_out_.data(), lw.gate_proj_w, num_new, intermediate_size, hidden_size);
        ops::matmul(ffn_up_.data(), norm_out_.data(), lw.up_proj_w, num_new, intermediate_size, hidden_size);

        ops::silu(ffn_gate_.data(), num_new * intermediate_size);
        ops::multiply(ffn_gate_.data(), ffn_gate_.data(), ffn_up_.data(), num_new * intermediate_size);
        ops::matmul(hidden_.data(), ffn_gate_.data(), lw.down_proj_w, num_new, hidden_size, intermediate_size);

        ops::add(hidden_.data(), hidden_.data(), residual_.data(), num_new * hidden_size);
    }

    // --- Final Norm + LM Head ---
    int last = (num_new - 1) * hidden_size;
    ops::rmsnorm(norm_out_.data(), hidden_.data() + last, weights_.final_norm, hidden_size, config_.rms_norm_eps);
    ops::matmul(logits, norm_out_.data(), weights_.lm_head, 1, config_.vocab_size, hidden_size);

    kv_cache_.seq_len = total_len;
}
