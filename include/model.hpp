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
    const float* q_proj_w = nullptr;
    const float* q_proj_b = nullptr;
    const float* k_proj_w = nullptr;
    const float* k_proj_b = nullptr;
    const float* v_proj_w = nullptr;
    const float* v_proj_b = nullptr;
    const float* o_proj_w = nullptr;

    const float* gate_proj_w = nullptr;
    const float* up_proj_w = nullptr;
    const float* down_proj_w = nullptr;

    const float* input_layernorm_w = nullptr;
    const float* post_attn_layernorm_w = nullptr;
};

struct ModelWeights {
    const float* embed_tokens = nullptr;
    const float* final_norm = nullptr;
    const float* lm_head = nullptr;
    std::vector<LayerWeights> layers;
};

struct KVCache {
    // Per-layer storage: [max_seq_len, num_kv_heads * head_dim]
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
    bool load(const std::string& model_dir);

    // Forward pass with KV-cache
    // First call (cache empty): prefill — processes all tokens, populates cache
    // Subsequent calls: decode — processes new tokens, appends to cache
    // Returns logits for the LAST token
    void forward(const std::vector<int>& tokens, float* logits);

    void reset();
    const ModelConfig& config() const;

private:
    ModelConfig config_;
    ModelWeights weights_;
    SafetensorsFile safetensors_;
    KVCache kv_cache_;

    // Activation buffers (sized for num_new tokens)
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

    void allocate_buffers(int num_new, int total_len);
    void load_weights();
};
