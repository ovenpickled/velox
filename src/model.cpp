#include "model.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "json.hpp"
#include "ops.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

// --- KVCache ---

// KVCache definitions are now inline in model.hpp

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
int Qwen2Model::max_context() const { return std::min(config_.max_position_embeddings, 4096); }

bool Qwen2Model::load(const std::string& model_dir, int max_batch_size) {
    config_ = ModelConfig::from_json(model_dir + "/config.json");

    if (config_.num_attention_heads <= 0 || config_.num_key_value_heads <= 0 ||
        config_.hidden_size % config_.num_attention_heads != 0 ||
        config_.num_attention_heads % config_.num_key_value_heads != 0) {
        std::cerr << "Invalid model config: hidden_size=" << config_.hidden_size
                  << " num_attention_heads=" << config_.num_attention_heads
                  << " num_key_value_heads=" << config_.num_key_value_heads << std::endl;
        return false;
    }
    if (max_batch_size < 1) {
        std::cerr << "max_batch_size must be at least 1" << std::endl;
        return false;
    }

    if (!safetensors_.open(model_dir + "/model.safetensors")) {
        std::cerr << "Failed to open safetensors file: " << model_dir << "/model.safetensors" << std::endl;
        return false;
    }

    if (!load_weights()) {
        std::cerr << "Failed to load model weights from " << model_dir << std::endl;
        return false;
    }

    max_batch_size_ = max_batch_size;
    kv_caches_.resize(max_batch_size);
    for (int b = 0; b < max_batch_size; b++) {
        kv_caches_[b].init(config_.num_hidden_layers, config_.num_key_value_heads, config_.head_dim, 128);
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

bool Qwen2Model::load_weights() {
    weights_.layers.resize(config_.num_hidden_layers);

    bool ok = true;
    auto fail = [&](const std::string& msg) {
        std::cerr << "[load] " << msg << std::endl;
        ok = false;
    };

    const int hidden = config_.hidden_size;
    const int q_dim = config_.num_attention_heads * config_.head_dim;
    const int kv_dim = config_.num_key_value_heads * config_.head_dim;
    const int inter = config_.intermediate_size;

    // Fetch a tensor and check that it exists, has a supported dtype, and has the expected
    // shape (d0 x d1, or just d0 for 1-D). Pass -1 to skip the shape check.
    auto fetch = [&](const std::string& name, int d0, int d1, Tensor& out) -> bool {
        if (!safetensors_.has_tensor(name)) { fail("missing tensor: " + name); return false; }
        out = safetensors_.get_tensor(name);
        if (out.empty()) { fail("could not read tensor (unsupported dtype or empty): " + name); return false; }
        if (d0 >= 0) {
            const int dims = (d1 >= 0) ? 2 : 1;
            const bool match = out.ndim() == dims && out.shape(0) == d0 && (d1 < 0 || out.shape(1) == d1);
            if (!match) {
                std::string got = "[";
                for (int i = 0; i < out.ndim(); i++) got += (i ? ", " : "") + std::to_string(out.shape(i));
                got += "]";
                fail("unexpected shape for " + name + ": got " + got + ", expected [" + std::to_string(d0) +
                     (d1 >= 0 ? ", " + std::to_string(d1) : std::string()) + "]");
                return false;
            }
        }
        return true;
    };

    auto copy_tensor = [&](std::vector<float>& dest, const std::string& name, int d0 = -1, int d1 = -1) {
        Tensor t;
        if (!fetch(name, d0, d1, t)) return;
        dest.assign(t.data(), t.data() + t.size());
    };

    auto quantize_tensor = [&](Int8Tensor& dest, const std::string& name, int d0 = -1, int d1 = -1) {
        Tensor t;
        if (!fetch(name, d0, d1, t)) return;
        if (t.ndim() != 2) { fail("expected a 2-D weight matrix: " + name); return; }
        // Safetensors shape is [N, K], so M = N (output dim), K = K (input dim)
        ops::quantize_rowwise(dest, t.data(), t.shape(0), t.shape(1));
    };

    // Qwen2 has q/k/v biases; if a checkpoint genuinely has none, use zeros instead of reading garbage
    auto load_bias = [&](std::vector<float>& dest, const std::string& name, int n) {
        if (!safetensors_.has_tensor(name)) { dest.assign(n, 0.0f); return; }
        copy_tensor(dest, name, n);
    };

    copy_tensor(weights_.final_norm, "model.norm.weight", hidden);

    // Embedding table: keep an exact FP32 copy for token lookups. When tied,
    // quantize the same data once for the output projection instead of
    // keeping a second FP32 copy.
    {
        Tensor t;
        if (fetch("model.embed_tokens.weight", config_.vocab_size, hidden, t)) {
            weights_.embed_tokens.assign(t.data(), t.data() + t.size());
            if (config_.tie_word_embeddings) {
                ops::quantize_rowwise(weights_.lm_head, t.data(), t.shape(0), t.shape(1));
            }
        }
    }
    if (!config_.tie_word_embeddings) {
        quantize_tensor(weights_.lm_head, "lm_head.weight", config_.vocab_size, hidden);
    }

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string prefix = "model.layers." + std::to_string(i);
        LayerWeights& lw = weights_.layers[i];

        quantize_tensor(lw.q_proj, prefix + ".self_attn.q_proj.weight", q_dim, hidden);
        load_bias(lw.q_proj_b, prefix + ".self_attn.q_proj.bias", q_dim);
        quantize_tensor(lw.k_proj, prefix + ".self_attn.k_proj.weight", kv_dim, hidden);
        load_bias(lw.k_proj_b, prefix + ".self_attn.k_proj.bias", kv_dim);
        quantize_tensor(lw.v_proj, prefix + ".self_attn.v_proj.weight", kv_dim, hidden);
        load_bias(lw.v_proj_b, prefix + ".self_attn.v_proj.bias", kv_dim);
        quantize_tensor(lw.o_proj, prefix + ".self_attn.o_proj.weight", hidden, q_dim);

        quantize_tensor(lw.gate_proj, prefix + ".mlp.gate_proj.weight", inter, hidden);
        quantize_tensor(lw.up_proj, prefix + ".mlp.up_proj.weight", inter, hidden);
        quantize_tensor(lw.down_proj, prefix + ".mlp.down_proj.weight", hidden, inter);

        copy_tensor(lw.input_layernorm_w, prefix + ".input_layernorm.weight", hidden);
        copy_tensor(lw.post_attn_layernorm_w, prefix + ".post_attention_layernorm.weight", hidden);
    }

    // Everything has been copied or quantized: release the FP32 conversions and unmap the file.
    safetensors_.close();
    stored_tensors_.clear();
    return ok;
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
    attn_scores_.resize(static_cast<size_t>(batch_size) * num_heads * max_len * max_total_len);
    ffn_gate_.resize(total_pos * intermediate_size);
    ffn_up_.resize(total_pos * intermediate_size);
}

// --- Single-sequence wrapper ---

void Qwen2Model::forward(const std::vector<int>& tokens, float* logits) {
    forward_batch({tokens}, logits);
}

// --- Batched Forward Pass ---

void Qwen2Model::forward_batch(const std::vector<std::vector<int>>& batch_tokens, float* logits) {
    auto t_fwd0 = std::chrono::steady_clock::now();
    int batch_size = batch_tokens.size();
    if (batch_size == 0) return;
    if (batch_size > max_batch_size_) {
        throw std::invalid_argument("forward_batch: batch size " + std::to_string(batch_size) +
                                    " exceeds the model's max batch size " + std::to_string(max_batch_size_));
    }
    static const bool prof_on = std::getenv("VELOX_PROF") != nullptr;   // per-step timing summary on stderr
    static const bool dbg_on = std::getenv("VELOX_DEBUG_ACT") != nullptr;
    static int dbg_calls = 0;
    const bool dbg_now = dbg_on && (dbg_calls++ == 0);   // first call (prefill) only
    auto dbg_stats = [&](int layer, const char* name, const float* x, int n, const char* where) {
        float mx = 0, sum = 0; int arg = 0;
        for (int i = 0; i < n; i++) {
            float v = std::fabs(x[i]);
            if (v > mx) { mx = v; arg = i; }
            sum += v;
        }
        float mean = sum / n;
        std::fprintf(stderr, "[act] L%02d %-8s %-5s max %9.2f  mean %8.4f  ratio %7.1f  ch %d\n",
                     layer, name, where, mx, mean, mx / (mean + 1e-9f), arg);
    };

    // Per-sequence metadata
    std::vector<int> actual_lens(batch_size);
    std::vector<int> past_lens(batch_size);
    int max_len = 0;
    int max_total_len = 0;

    for (int b = 0; b < batch_size; b++) {
        actual_lens[b] = batch_tokens[b].size();
        past_lens[b] = kv_caches_[b].seq_len;
        if (actual_lens[b] == 0) {
            throw std::invalid_argument("forward_batch: empty token list for sequence " + std::to_string(b));
        }
        for (int tok : batch_tokens[b]) {
            if (tok < 0 || tok >= config_.vocab_size) {
                throw std::invalid_argument("forward_batch: token id " + std::to_string(tok) + " is outside the vocabulary");
            }
        }

        // Refuse to run past the KV cache limit instead of overflowing the cache buffers
        int needed_len = past_lens[b] + actual_lens[b];
        const int cache_max = max_context();
        if (needed_len > cache_max) {
            throw std::runtime_error("sequence length " + std::to_string(needed_len) +
                                     " exceeds the KV cache limit of " + std::to_string(cache_max) + " tokens");
        }
        kv_caches_[b].ensure_capacity(needed_len);
        
        max_len = std::max(max_len, actual_lens[b]);
        max_total_len = std::max(max_total_len, needed_len);
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
        #pragma omp parallel for
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
        #pragma omp parallel for
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
        #pragma omp parallel for collapse(2) if(static_cast<size_t>(batch_size) * num_heads >= 16)
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                int al = actual_lens[b];
                int pl = past_lens[b];
                int tl = pl + al;
                float* k_cache = kv_caches_[b].k_cache[layer].data();
                float* v_cache = kv_caches_[b].v_cache[layer].data();
                int kv_h = h / heads_per_group;

                for (int qp = 0; qp < al; qp++) {
                    int gp = b * max_len + qp;
                    int actual_qpos = pl + qp;
                    float* scores = attn_scores_.data() 
                        + ((static_cast<size_t>(b) * num_heads + h) * max_len + qp) * max_total_len;

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

        #pragma omp parallel for
        for (int p = 0; p < total_positions; p++) {
            ops::rmsnorm(norm_out_.data() + p * hidden_size,
                        hidden_.data() + p * hidden_size,
                        lw.post_attn_layernorm_w.data(), hidden_size, config_.rms_norm_eps);
        }

        ops::matmul_w8a8(ffn_gate_.data(), norm_out_.data(), lw.gate_proj, total_positions, intermediate_size, hidden_size);
        ops::matmul_w8a8(ffn_up_.data(), norm_out_.data(), lw.up_proj, total_positions, intermediate_size, hidden_size);

        ops::silu(ffn_gate_.data(), total_positions * intermediate_size);
        ops::multiply(ffn_gate_.data(), ffn_gate_.data(), ffn_up_.data(), total_positions * intermediate_size);
        if (dbg_now) {
            int last = actual_lens[0] - 1;
            dbg_stats(layer, "down_in", ffn_gate_.data() + 0 * intermediate_size, intermediate_size, "pos0");
            dbg_stats(layer, "down_in", ffn_gate_.data() + last * intermediate_size, intermediate_size, "last");
        }
        ops::matmul_w8a8(hidden_.data(), ffn_gate_.data(), lw.down_proj, total_positions, hidden_size, intermediate_size);

        ops::add(hidden_.data(), hidden_.data(), residual_.data(), total_positions * hidden_size);
        if (dbg_now) {
            int last = actual_lens[0] - 1;
            dbg_stats(layer, "hidden", hidden_.data() + 0 * hidden_size, hidden_size, "pos0");
            dbg_stats(layer, "hidden", hidden_.data() + last * hidden_size, hidden_size, "last");
        }
    }

    // --- Final Norm + LM Head (batched) ---
    // Extract last valid position per sequence, then batched projection
    for (int b = 0; b < batch_size; b++) {
        int last_pos = b * max_len + actual_lens[b] - 1;
        ops::rmsnorm(norm_out_.data() + b * hidden_size,
                    hidden_.data() + last_pos * hidden_size,
                    weights_.final_norm.data(), hidden_size, config_.rms_norm_eps);
    }
    auto t_lm0 = std::chrono::steady_clock::now();
    ops::matmul_w8a8(logits, norm_out_.data(), weights_.lm_head, batch_size, config_.vocab_size, config_.hidden_size);
    auto t_lm1 = std::chrono::steady_clock::now();

    // Update cache positions
    for (int b = 0; b < batch_size; b++) {
        kv_caches_[b].seq_len = past_lens[b] + actual_lens[b];
    }
    auto t_fwd1 = std::chrono::steady_clock::now();
    if (prof_on) {
        static double lm_ms = 0, fwd_ms = 0; static int n = 0;
        lm_ms  += std::chrono::duration<double, std::milli>(t_lm1  - t_lm0 ).count();
        fwd_ms += std::chrono::duration<double, std::milli>(t_fwd1 - t_fwd0).count();
        if (++n % 32 == 0)
            std::fprintf(stderr, "[prof] bs=%d  lm_head %.1f ms  forward %.1f ms  (%.0f%%)\n",
                         batch_size, lm_ms / n, fwd_ms / n, 100.0 * lm_ms / fwd_ms);
    }
}
