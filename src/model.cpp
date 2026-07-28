#include "model.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
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

const ModelConfig& Qwen2Model::config() const { return config_; }
int Qwen2Model::max_batch_size() const { return max_batch_size_; }

bool Qwen2Model::load(const std::string& model_dir, int max_batch_size) {
    config_ = ModelConfig::from_json(model_dir + "/config.json");

    if (!safetensors_.open(model_dir + "/model.safetensors")) {
        std::cerr << "Failed to open safetensors file: " << model_dir << "/model.safetensors" << std::endl;
        return false;
    }

    load_weights();

    max_batch_size_ = max_batch_size;
    int cache_max = std::min(config_.max_position_embeddings, 4096);
    kv_caches_.resize(max_batch_size);
    for (int b = 0; b < max_batch_size; b++) {
        kv_caches_[b].init(config_.num_hidden_layers, config_.num_key_value_heads, config_.head_dim, cache_max);
    }

    allocate_buffers(1, 1, 1);
    return true;
}

void Qwen2Model::reset() {
    for (auto& cache : kv_caches_) cache.reset();
}

void Qwen2Model::reset(int batch_idx) {
    kv_caches_[batch_idx].reset();
}

void Qwen2Model::load_weights() {
    weights_.layers.resize(config_.num_hidden_layers);

    auto copy_tensor = [&](std::vector<float>& dest, const std::string& name) {
        if (!safetensors_.has_tensor(name)) return;
        Tensor t = safetensors_.get_tensor(name);
        dest.assign(t.data(), t.data() + t.size());
    };

    auto quantize_tensor = [&](Int8Tensor& dest, const std::string& name) {
        if (!safetensors_.has_tensor(name)) return;
        Tensor t = safetensors_.get_tensor(name);
        // Safetensors shape is [N, K], so M = N (output dim), K = K (input dim)
        ops::quantize_rowwise(dest, t.data(), t.shape(0), t.shape(1));
    };

    copy_tensor(weights_.embed_tokens, "model.embed_tokens.weight");
    copy_tensor(weights_.final_norm, "model.norm.weight");

    if (config_.tie_word_embeddings) {
        weights_.lm_head = weights_.embed_tokens;
    } else {
        copy_tensor(weights_.lm_head, "lm_head.weight");
    }

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string prefix = "model.layers." + std::to_string(i);
        LayerWeights& lw = weights_.layers[i];

        quantize_tensor(lw.q_proj, prefix + ".self_attn.q_proj.weight");
        copy_tensor(lw.q_proj_b, prefix + ".self_attn.q_proj.bias");
        quantize_tensor(lw.k_proj, prefix + ".self_attn.k_proj.weight");
        copy_tensor(lw.k_proj_b, prefix + ".self_attn.k_proj.bias");
        quantize_tensor(lw.v_proj, prefix + ".self_attn.v_proj.weight");
        copy_tensor(lw.v_proj_b, prefix + ".self_attn.v_proj.bias");
        quantize_tensor(lw.o_proj, prefix + ".self_attn.o_proj.weight");

        quantize_tensor(lw.gate_proj, prefix + ".mlp.gate_proj.weight");
        quantize_tensor(lw.up_proj, prefix + ".mlp.up_proj.weight");
        quantize_tensor(lw.down_proj, prefix + ".mlp.down_proj.weight");

        copy_tensor(lw.input_layernorm_w, prefix + ".input_layernorm.weight");
        copy_tensor(lw.post_attn_layernorm_w, prefix + ".post_attention_layernorm.weight");
    }
    
    // Free the FP32 buffers from memory!
    safetensors_.clear();
    stored_tensors_.clear();
}

void Qwen2Model::allocate_buffers(int batch_size, int max_len, int max_total_len) {
    int hidden_size = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim;
    int intermediate_size = config_.intermediate_size;
    size_t total_pos = static_cast<size_t>(batch_size) * max_len;

    hidden_.resize(total_pos * hidden_size);
    residual_.resize(total_pos * hidden_size);
    norm_out_.resize(std::max(total_pos * hidden_size, static_cast<size_t>(batch_size) * hidden_size));
    q_buf_.resize(total_pos * num_heads * head_dim);
    k_buf_.resize(total_pos * num_kv_heads * head_dim);
    v_buf_.resize(total_pos * num_kv_heads * head_dim);
    attn_out_.resize(total_pos * num_heads * head_dim);
    attn_scores_.resize(static_cast<size_t>(num_heads) * max_len * max_total_len);
    ffn_gate_.resize(total_pos * intermediate_size);
    ffn_up_.resize(total_pos * intermediate_size);
}

// --- Single-sequence wrapper ---

void Qwen2Model::forward(const std::vector<int>& tokens, float* logits) {
    forward_batch({tokens}, logits);
}

// --- Batched Forward Pass ---

void Qwen2Model::forward_batch(const std::vector<std::vector<int>>& batch_tokens, float* logits) {
    int batch_size = batch_tokens.size();

    // Per-sequence metadata
    std::vector<int> actual_lens(batch_size);
    std::vector<int> past_lens(batch_size);
    int max_len = 0;
    int max_total_len = 0;

    for (int b = 0; b < batch_size; b++) {
        actual_lens[b] = batch_tokens[b].size();
        past_lens[b] = kv_caches_[b].seq_len;
        max_len = std::max(max_len, actual_lens[b]);
        max_total_len = std::max(max_total_len, past_lens[b] + actual_lens[b]);
    }

    int total_positions = batch_size * max_len;
    allocate_buffers(batch_size, max_len, max_total_len);

    int hidden_size = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_key_value_heads;
    int head_dim = config_.head_dim;
    int intermediate_size = config_.intermediate_size;
    int kv_dim = num_kv_heads * head_dim;
    int q_dim = num_heads * head_dim;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int heads_per_group = num_heads / num_kv_heads;

    // --- Embedding (with zero-padding for short sequences) ---
    std::fill(hidden_.begin(), hidden_.begin() + total_positions * hidden_size, 0.0f);
    for (int b = 0; b < batch_size; b++) {
        for (int p = 0; p < actual_lens[b]; p++) {
            ops::embedding_lookup(hidden_.data() + (b * max_len + p) * hidden_size,
                                 weights_.embed_tokens.data(), batch_tokens[b][p], hidden_size);
        }
    }

    // --- Transformer Layers ---
    for (int layer = 0; layer < config_.num_hidden_layers; layer++) {
        LayerWeights& lw = weights_.layers[layer];

        // Residual
        std::memcpy(residual_.data(), hidden_.data(), total_positions * hidden_size * sizeof(float));

        // Input LayerNorm (all positions including padding — zero input produces zero output)
        for (int p = 0; p < total_positions; p++) {
            ops::rmsnorm(norm_out_.data() + p * hidden_size,
                        hidden_.data() + p * hidden_size,
                        lw.input_layernorm_w.data(), hidden_size, config_.rms_norm_eps);
        }

        // Q/K/V Projections (batched W8A8 matmul: M = total_positions)
        ops::matmul_w8a8(q_buf_.data(), norm_out_.data(), lw.q_proj, total_positions, q_dim, hidden_size);
        ops::matmul_w8a8(k_buf_.data(), norm_out_.data(), lw.k_proj, total_positions, kv_dim, hidden_size);
        ops::matmul_w8a8(v_buf_.data(), norm_out_.data(), lw.v_proj, total_positions, kv_dim, hidden_size);

        // Bias (all positions)
        for (int p = 0; p < total_positions; p++) {
            ops::add_bias(q_buf_.data() + p * q_dim, lw.q_proj_b.data(), q_dim);
            ops::add_bias(k_buf_.data() + p * kv_dim, lw.k_proj_b.data(), kv_dim);
            ops::add_bias(v_buf_.data() + p * kv_dim, lw.v_proj_b.data(), kv_dim);
        }

        // RoPE (per-sequence positions, skip padding)
        for (int b = 0; b < batch_size; b++) {
            for (int p = 0; p < actual_lens[b]; p++) {
                int gp = b * max_len + p;
                ops::rope(q_buf_.data() + gp * q_dim,
                         k_buf_.data() + gp * kv_dim,
                         head_dim, num_heads, num_kv_heads,
                         past_lens[b] + p, config_.rope_theta);
            }
        }

        // Append K/V to per-sequence caches (valid positions only)
        for (int b = 0; b < batch_size; b++) {
            float* k_cache = kv_caches_[b].k_cache[layer].data();
            float* v_cache = kv_caches_[b].v_cache[layer].data();
            std::memcpy(k_cache + past_lens[b] * kv_dim,
                       k_buf_.data() + b * max_len * kv_dim,
                       actual_lens[b] * kv_dim * sizeof(float));
            std::memcpy(v_cache + past_lens[b] * kv_dim,
                       v_buf_.data() + b * max_len * kv_dim,
                       actual_lens[b] * kv_dim * sizeof(float));
        }

        // Attention (per-sequence — different cache lengths)
        for (int b = 0; b < batch_size; b++) {
            int al = actual_lens[b];
            int pl = past_lens[b];
            int tl = pl + al;
            float* k_cache = kv_caches_[b].k_cache[layer].data();
            float* v_cache = kv_caches_[b].v_cache[layer].data();

            for (int h = 0; h < num_heads; h++) {
                int kv_h = h / heads_per_group;

                for (int qp = 0; qp < al; qp++) {
                    int gp = b * max_len + qp;
                    int actual_qpos = pl + qp;
                    float* scores = attn_scores_.data() + (h * al + qp) * tl;

                    for (int kp = 0; kp < tl; kp++) {
                        if (kp > actual_qpos) {
                            scores[kp] = -INFINITY;
                        } else {
                            float dot = 0;
                            for (int d = 0; d < head_dim; d++) {
                                dot += q_buf_[gp * q_dim + h * head_dim + d]
                                     * k_cache[kp * kv_dim + kv_h * head_dim + d];
                            }
                            scores[kp] = dot * scale;
                        }
                    }
                    ops::softmax(scores, tl);

                    for (int d = 0; d < head_dim; d++) {
                        float val = 0;
                        for (int kp = 0; kp < tl; kp++) {
                            val += scores[kp] * v_cache[kp * kv_dim + kv_h * head_dim + d];
                        }
                        attn_out_[gp * q_dim + h * head_dim + d] = val;
                    }
                }
            }
        }

        // O Projection + Residual (batched W8A8 matmul)
        ops::matmul_w8a8(hidden_.data(), attn_out_.data(), lw.o_proj, total_positions, hidden_size, q_dim);
        ops::add(hidden_.data(), hidden_.data(), residual_.data(), total_positions * hidden_size);

        // FFN
        std::memcpy(residual_.data(), hidden_.data(), total_positions * hidden_size * sizeof(float));

        for (int p = 0; p < total_positions; p++) {
            ops::rmsnorm(norm_out_.data() + p * hidden_size,
                        hidden_.data() + p * hidden_size,
                        lw.post_attn_layernorm_w.data(), hidden_size, config_.rms_norm_eps);
        }

        ops::matmul_w8a8(ffn_gate_.data(), norm_out_.data(), lw.gate_proj, total_positions, intermediate_size, hidden_size);
        ops::matmul_w8a8(ffn_up_.data(), norm_out_.data(), lw.up_proj, total_positions, intermediate_size, hidden_size);

        ops::silu(ffn_gate_.data(), total_positions * intermediate_size);
        ops::multiply(ffn_gate_.data(), ffn_gate_.data(), ffn_up_.data(), total_positions * intermediate_size);
        ops::matmul_w8a8(hidden_.data(), ffn_gate_.data(), lw.down_proj, total_positions, hidden_size, intermediate_size);

        ops::add(hidden_.data(), hidden_.data(), residual_.data(), total_positions * hidden_size);
    }

    // --- Final Norm + LM Head (batched) ---
    // Extract last valid position per sequence, then batched projection
    for (int b = 0; b < batch_size; b++) {
        int last_pos = b * max_len + actual_lens[b] - 1;
        ops::rmsnorm(norm_out_.data() + b * hidden_size,
                    hidden_.data() + last_pos * hidden_size,
                    weights_.final_norm.data(), hidden_size, config_.rms_norm_eps);
    }
    ops::matmul(logits, norm_out_.data(), weights_.lm_head.data(), batch_size, config_.vocab_size, hidden_size);

    // Update cache positions
    for (int b = 0; b < batch_size; b++) {
        kv_caches_[b].seq_len = past_lens[b] + actual_lens[b];
    }
}
