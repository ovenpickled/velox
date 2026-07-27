#include "model.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include "json.hpp"
#include "ops.hpp"

using json = nlohmann::json;

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
    allocate_buffers(1);
    return true;
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

void Qwen2Model::allocate_buffers(int seq_len) {
    int hidden_size = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim;
    int intermediate_size = config_.intermediate_size;

    hidden_.resize(seq_len * hidden_size);
    residual_.resize(seq_len * hidden_size);
    norm_out_.resize(seq_len * hidden_size);
    q_buf_.resize(seq_len * num_heads * head_dim);
    k_buf_.resize(seq_len * num_kv_heads * head_dim);
    v_buf_.resize(seq_len * num_kv_heads * head_dim);
    attn_out_.resize(seq_len * num_heads * head_dim);
    attn_scores_.resize(num_heads * seq_len * seq_len);
    ffn_gate_.resize(seq_len * intermediate_size);
    ffn_up_.resize(seq_len * intermediate_size);
    ffn_down_.resize(seq_len * hidden_size);
}

void Qwen2Model::forward(const std::vector<int>& tokens, float* logits) {
    int seq_len = tokens.size();
    allocate_buffers(seq_len);

    int hidden_size = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim;
    int intermediate_size = config_.intermediate_size;
    float rms_norm_eps = config_.rms_norm_eps;
    float rope_theta = config_.rope_theta;

    for (int p = 0; p < seq_len; p++) {
        ops::embedding_lookup(hidden_.data() + p * hidden_size, weights_.embed_tokens, tokens[p], hidden_size);
    }

    for (int layer = 0; layer < config_.num_hidden_layers; layer++) {
        LayerWeights& lw = weights_.layers[layer];
        
        std::memcpy(residual_.data(), hidden_.data(), seq_len * hidden_size * sizeof(float));
        
        for (int p = 0; p < seq_len; p++) {
            ops::rmsnorm(norm_out_.data() + p * hidden_size, hidden_.data() + p * hidden_size, lw.input_layernorm_w, hidden_size, rms_norm_eps);
        }
        
        ops::matmul(q_buf_.data(), norm_out_.data(), lw.q_proj_w, seq_len, num_heads * head_dim, hidden_size);
        ops::matmul(k_buf_.data(), norm_out_.data(), lw.k_proj_w, seq_len, num_kv_heads * head_dim, hidden_size);
        ops::matmul(v_buf_.data(), norm_out_.data(), lw.v_proj_w, seq_len, num_kv_heads * head_dim, hidden_size);
        
        for (int p = 0; p < seq_len; p++) {
            ops::add_bias(q_buf_.data() + p * num_heads * head_dim, lw.q_proj_b, num_heads * head_dim);
            ops::add_bias(k_buf_.data() + p * num_kv_heads * head_dim, lw.k_proj_b, num_kv_heads * head_dim);
            ops::add_bias(v_buf_.data() + p * num_kv_heads * head_dim, lw.v_proj_b, num_kv_heads * head_dim);
        }
        
        for (int p = 0; p < seq_len; p++) {
            ops::rope(q_buf_.data() + p * num_heads * head_dim, k_buf_.data() + p * num_kv_heads * head_dim, head_dim, num_heads, num_kv_heads, p, rope_theta);
        }
        
        int heads_per_group = num_heads / num_kv_heads;
        
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_group;
            
            for (int query_pos = 0; query_pos < seq_len; query_pos++) {
                for (int key_pos = 0; key_pos < seq_len; key_pos++) {
                    if (key_pos > query_pos) {
                        attn_scores_[h * seq_len * seq_len + query_pos * seq_len + key_pos] = -INFINITY;
                    } else {
                        float score = 0;
                        for (int d = 0; d < head_dim; d++) {
                            float q = q_buf_[query_pos * num_heads * head_dim + h * head_dim + d];
                            float k = k_buf_[key_pos * num_kv_heads * head_dim + kv_h * head_dim + d];
                            score += q * k;
                        }
                        attn_scores_[h * seq_len * seq_len + query_pos * seq_len + key_pos] = score / std::sqrt(static_cast<float>(head_dim));
                    }
                }
                ops::softmax(attn_scores_.data() + h * seq_len * seq_len + query_pos * seq_len, seq_len);
            }
            
            for (int query_pos = 0; query_pos < seq_len; query_pos++) {
                for (int d = 0; d < head_dim; d++) {
                    float val = 0;
                    for (int key_pos = 0; key_pos < seq_len; key_pos++) {
                        float v = v_buf_[key_pos * num_kv_heads * head_dim + kv_h * head_dim + d];
                        val += attn_scores_[h * seq_len * seq_len + query_pos * seq_len + key_pos] * v;
                    }
                    attn_out_[query_pos * num_heads * head_dim + h * head_dim + d] = val;
                }
            }
        }
        
        ops::matmul(hidden_.data(), attn_out_.data(), lw.o_proj_w, seq_len, hidden_size, num_heads * head_dim);
        ops::add(hidden_.data(), hidden_.data(), residual_.data(), seq_len * hidden_size);
        
        std::memcpy(residual_.data(), hidden_.data(), seq_len * hidden_size * sizeof(float));
        
        for (int p = 0; p < seq_len; p++) {
            ops::rmsnorm(norm_out_.data() + p * hidden_size, hidden_.data() + p * hidden_size, lw.post_attn_layernorm_w, hidden_size, rms_norm_eps);
        }
        
        ops::matmul(ffn_gate_.data(), norm_out_.data(), lw.gate_proj_w, seq_len, intermediate_size, hidden_size);
        ops::matmul(ffn_up_.data(), norm_out_.data(), lw.up_proj_w, seq_len, intermediate_size, hidden_size);
        
        ops::silu(ffn_gate_.data(), seq_len * intermediate_size);
        ops::multiply(ffn_gate_.data(), ffn_gate_.data(), ffn_up_.data(), seq_len * intermediate_size);
        ops::matmul(hidden_.data(), ffn_gate_.data(), lw.down_proj_w, seq_len, hidden_size, intermediate_size);
        
        ops::add(hidden_.data(), hidden_.data(), residual_.data(), seq_len * hidden_size);
    }
    
    int last = (seq_len - 1) * hidden_size;
    ops::rmsnorm(norm_out_.data(), hidden_.data() + last, weights_.final_norm, hidden_size, rms_norm_eps);
    ops::matmul(logits, norm_out_.data(), weights_.lm_head, 1, config_.vocab_size, hidden_size);
}
