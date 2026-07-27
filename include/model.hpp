// model.hpp
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
    const float* lm_head = nullptr;  // may point to embed_tokens if tied
    std::vector<LayerWeights> layers;
};

class Qwen2Model {
public:
    bool load(const std::string& model_dir);

    // Forward pass: takes a sequence of token IDs, returns logits for the LAST token
    // logits is a float array of size vocab_size
    void forward(const std::vector<int>& tokens, float* logits);

    const ModelConfig& config() const;

private:
    ModelConfig config_;
    ModelWeights weights_;
    SafetensorsFile safetensors_;

    // Pre-allocated activation buffers
    std::vector<float> hidden_;       // [seq_len * hidden_size]
    std::vector<float> residual_;     // [seq_len * hidden_size]
    std::vector<float> norm_out_;     // [seq_len * hidden_size]
    std::vector<float> q_buf_;        // [seq_len * num_heads * head_dim]
    std::vector<float> k_buf_;        // [seq_len * num_kv_heads * head_dim]
    std::vector<float> v_buf_;        // [seq_len * num_kv_heads * head_dim]
    std::vector<float> attn_out_;     // [seq_len * num_heads * head_dim]
    std::vector<float> attn_scores_;  // [num_heads * seq_len * seq_len]
    std::vector<float> ffn_gate_;     // [seq_len * intermediate_size]
    std::vector<float> ffn_up_;       // [seq_len * intermediate_size]
    std::vector<float> ffn_down_;     // [seq_len * hidden_size]

    // Stored tensors from safetensors (keeps mmap and converted data alive)
    std::vector<Tensor> stored_tensors_;

    void allocate_buffers(int seq_len);
    void load_weights();
};
