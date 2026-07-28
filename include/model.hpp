#pragma once
#include <string>
#include <vector>
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
};

struct ModelWeights {
    std::vector<float> embed_tokens;
    std::vector<float> final_norm;
    std::vector<float> lm_head;
    std::vector<LayerWeights> layers;
};

struct KVCache {
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    int seq_len = 0;
    int max_seq_len = 0;
    int kv_dim = 0;

    void init(int num_layers, int num_kv_heads, int head_dim, int max_seq_len);
    void reset();
};

class Qwen2Model {
public:
    bool load(const std::string& model_dir, int max_batch_size = 1);

    // Single sequence (convenience wrapper, uses batch slot 0)
    void forward(const std::vector<int>& tokens, float* logits);

    // Batched: processes multiple sequences simultaneously
    // batch_tokens[b] = new tokens for sequence b
    // logits output: [batch_size, vocab_size]
    void forward_batch(const std::vector<std::vector<int>>& batch_tokens, float* logits);

    void reset();
    void reset(int batch_idx);
    int max_batch_size() const;
    const ModelConfig& config() const;

private:
    ModelConfig config_;
    ModelWeights weights_;
    SafetensorsFile safetensors_;

    std::vector<KVCache> kv_caches_;
    int max_batch_size_ = 1;

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

    void allocate_buffers(int batch_size, int max_len, int max_total_len);
    void load_weights();
};
