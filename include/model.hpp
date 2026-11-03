#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "tensor.hpp"
#include "safetensors.hpp"

struct ModelConfig {
    int hidden_size = 896;
    int num_hidden_layers = 24;
    int num_attention_heads = 14;
    int num_key_value_heads = 2;
    int head_dim = 64;
    int intermediate_size = 4864;
    int vocab_size = 151936;
    int max_position_embeddings = 32768;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    bool tie_word_embeddings = true;

    static ModelConfig from_json(const std::string& config_json_path);
};

struct LayerWeights {
    Int8Tensor q_proj;
    Int8Tensor k_proj;
    Int8Tensor v_proj;
    Int8Tensor o_proj;

    Int8Tensor gate_proj;
    Int8Tensor up_proj;
    Int8Tensor down_proj;

    std::vector<float> q_proj_b;
    std::vector<float> k_proj_b;
    std::vector<float> v_proj_b;

    std::vector<float> input_layernorm_w;
    std::vector<float> post_attn_layernorm_w;

    // Diagnostic: FP32 copies of the projection weights for the layers selected with
    // VELOX_FP32_LAYERS. Empty means "use the INT8 path".
    std::vector<float> q_f32, k_f32, v_f32, o_f32, gate_f32, up_f32, down_f32;
};

struct ModelWeights {
    std::vector<float> embed_tokens;
    std::vector<float> final_norm;
    Int8Tensor lm_head;
    std::vector<LayerWeights> layers;
};

enum class KVType {
    FP32,
    INT8
};

struct KVCache {
    KVType type = KVType::FP32;
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;

    // INT8 KV cache: symmetric per-token per-head INT8 quantization
    std::vector<std::vector<int8_t>> k_cache_i8;
    std::vector<std::vector<float>> k_scales;
    std::vector<std::vector<int8_t>> v_cache_i8;
    std::vector<std::vector<float>> v_scales;

    int seq_len = 0;
    int capacity = 0;
    int kv_dim = 0;
    int num_kv_heads = 0;
    int head_dim = 0;

    void init(int num_layers, int n_kv_heads, int h_dim, int initial_capacity = 256, KVType kv_type = KVType::FP32) {
        num_kv_heads = n_kv_heads;
        head_dim = h_dim;
        kv_dim = num_kv_heads * head_dim;
        capacity = initial_capacity;
        seq_len = 0;
        type = kv_type;
        size_t layer_size = static_cast<size_t>(capacity) * kv_dim;
        size_t scale_size = static_cast<size_t>(capacity) * num_kv_heads;

        if (type == KVType::FP32) {
            k_cache.resize(num_layers);
            v_cache.resize(num_layers);
            for (int i = 0; i < num_layers; i++) {
                k_cache[i].resize(layer_size, 0.0f);
                v_cache[i].resize(layer_size, 0.0f);
            }
        } else {
            k_cache_i8.resize(num_layers);
            k_scales.resize(num_layers);
            v_cache_i8.resize(num_layers);
            v_scales.resize(num_layers);
            for (int i = 0; i < num_layers; i++) {
                k_cache_i8[i].resize(layer_size, 0);
                k_scales[i].resize(scale_size, 0.0f);
                v_cache_i8[i].resize(layer_size, 0);
                v_scales[i].resize(scale_size, 0.0f);
            }
        }
    }

    void ensure_capacity(int needed) {
        if (needed <= capacity) return;
        int new_capacity = capacity;
        while (new_capacity < needed) new_capacity *= 2;
#ifdef DEBUG_KV_CACHE
        std::cout << "[DEBUG] KV Cache resizing from " << capacity << " to " << new_capacity << " (needed: " << needed << ")" << std::endl;
#endif
        size_t layer_size = static_cast<size_t>(new_capacity) * kv_dim;
        size_t scale_size = static_cast<size_t>(new_capacity) * num_kv_heads;

        if (type == KVType::FP32) {
            for (auto& layer : k_cache) layer.resize(layer_size, 0.0f);
            for (auto& layer : v_cache) layer.resize(layer_size, 0.0f);
        } else {
            for (auto& layer : k_cache_i8) layer.resize(layer_size, 0);
            for (auto& layer : k_scales) layer.resize(scale_size, 0.0f);
            for (auto& layer : v_cache_i8) layer.resize(layer_size, 0);
            for (auto& layer : v_scales) layer.resize(scale_size, 0.0f);
        }
        capacity = new_capacity;
    }

    void reset() {
        seq_len = 0;
    }

    size_t memory_bytes() const {
        size_t total = 0;
        if (type == KVType::FP32) {
            for (const auto& l : k_cache) total += l.capacity() * sizeof(float);
            for (const auto& l : v_cache) total += l.capacity() * sizeof(float);
        } else {
            for (const auto& l : k_cache_i8) total += l.capacity() * sizeof(int8_t);
            for (const auto& l : k_scales) total += l.capacity() * sizeof(float);
            for (const auto& l : v_cache_i8) total += l.capacity() * sizeof(int8_t);
            for (const auto& l : v_scales) total += l.capacity() * sizeof(float);
        }
        return total;
    }

    size_t active_bytes() const {
        if (seq_len <= 0) return 0;
        size_t num_layers = (type == KVType::FP32) ? k_cache.size() : k_cache_i8.size();
        if (type == KVType::FP32) {
            return num_layers * static_cast<size_t>(seq_len) * kv_dim * sizeof(float) * 2;
        } else {
            size_t bytes_per_pos = kv_dim * sizeof(int8_t) + num_kv_heads * sizeof(float);
            return num_layers * static_cast<size_t>(seq_len) * bytes_per_pos * 2;
        }
    }
};

class Qwen2Model {
public:
    bool load(const std::string& model_dir, int max_batch_size = 1, KVType kv_type = KVType::FP32);

    // Single sequence (convenience wrapper, uses batch slot 0)
    void forward(const std::vector<int>& tokens, float* logits);

    // Batched: processes multiple sequences simultaneously
    // batch_tokens[b] = new tokens for sequence b
    // logits output: [batch_size, vocab_size]
    void forward_batch(const std::vector<std::vector<int>>& batch_tokens, float* logits);

    void reset();
    void reset(int batch_idx);
    int max_batch_size() const;
    // Longest sequence (prompt + generated tokens) the KV cache will accept
    int max_context() const;
    const ModelConfig& config() const;
    KVType kv_type() const { return kv_type_; }
    size_t kv_cache_memory_bytes() const;
    size_t kv_cache_active_bytes() const;

private:
    ModelConfig config_;
    ModelWeights weights_;
    SafetensorsFile safetensors_;

    std::vector<KVCache> kv_caches_;
    int max_batch_size_ = 1;
    KVType kv_type_ = KVType::FP32;

    // Activation buffers (sized for batch_size * max_len)
    std::vector<float> hidden_;
    std::vector<float> residual_;
    std::vector<float> norm_out_;
    std::vector<float> q_buf_;
    std::vector<float> k_buf_;
    std::vector<float> v_buf_;
    std::vector<float> attn_out_;
    std::vector<float> attn_scores_;
    std::vector<float> ffn_gate_;
    std::vector<float> ffn_up_;

    std::vector<Tensor> stored_tensors_;

    // Load-time calibration (see load_weights): per-layer first-token activation maxima
    bool calibrating_ = false;
    std::vector<float> calib_ffn_max_, calib_down_max_;

    void allocate_buffers(int batch_size, int max_len, int max_total_len);
    bool load_weights();
};
