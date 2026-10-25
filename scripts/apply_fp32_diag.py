#!/usr/bin/env python3
"""Adds the VELOX_FP32_LAYERS diagnostic to model.hpp / model.cpp (safe to run twice).

Switches added (all off by default; with none set the engine behaves exactly as before):
  VELOX_FP32_LAYERS=all | 2,3,21   keep FP32 copies of those layers' weights and run them in FP32
  VELOX_FP32_LMHEAD=1              run the LM head in FP32 using the (tied) FP32 embedding table
  VELOX_W8A32=1                    FP32 activations everywhere the INT8 weights are still used
Everything FP32 = VELOX_FP32_LAYERS=all VELOX_FP32_LMHEAD=1: should match Hugging Face tightly.

usage:   python3 apply_fp32_diag.py [project_root]     (default: current directory)
Undo:    git checkout include/model.hpp src/model.cpp
"""
import os
import sys

if len(sys.argv) > 1:
    root = sys.argv[1]
elif os.path.exists(os.path.join(os.getcwd(), "src", "model.cpp")):
    root = os.getcwd()
else:
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
hpp_path = os.path.join(root, "include", "model.hpp")
cpp_path = os.path.join(root, "src", "model.cpp")


def replace_once(text, old, new, what):
    if text.count(old) != 1:
        sys.exit(f"cannot apply: expected exactly one match for {what}, found {text.count(old)}. "
                 f"Nothing was changed in this file.")
    return text.replace(old, new, 1)


if not (os.path.exists(hpp_path) and os.path.exists(cpp_path)):
    sys.exit(f"cannot find include/model.hpp and src/model.cpp under {os.path.abspath(root)}; run from the project root or pass it as an argument")

hpp = open(hpp_path).read()
cpp = open(cpp_path).read()

if "fp32_flags" in cpp or "q_f32" in hpp:
    sys.exit("already applied - nothing to do")

# ---- model.hpp: FP32 weight copies ----
hpp = replace_once(hpp,
    "    std::vector<float> post_attn_layernorm_w;\n};",
    "    std::vector<float> post_attn_layernorm_w;\n\n"
    "    // Diagnostic: FP32 copies of the projection weights for the layers selected with\n"
    "    // VELOX_FP32_LAYERS. Empty means \"use the INT8 path\".\n"
    "    std::vector<float> q_f32, k_f32, v_f32, o_f32, gate_f32, up_f32, down_f32;\n};",
    "end of struct LayerWeights")

# ---- model.cpp ----
cpp = replace_once(cpp, "#include <string>\n", "#include <string>\n#include <sstream>\n", "#include <string>")

cpp = replace_once(cpp,
    '    copy_tensor(weights_.final_norm, "model.norm.weight", hidden);\n',
    '''    // Diagnostic: VELOX_FP32_LAYERS=all, or a comma-separated list such as 2,3,21, keeps FP32
    // copies of those layers' weights and runs them without INT8 quantization.
    std::vector<bool> fp32_flags(config_.num_hidden_layers, false);
    if (const char* env = std::getenv("VELOX_FP32_LAYERS")) {
        std::string spec(env);
        try {
            if (spec == "all") {
                std::fill(fp32_flags.begin(), fp32_flags.end(), true);
            } else {
                std::stringstream ss(spec);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    int l = std::stoi(tok);
                    if (l < 0 || l >= config_.num_hidden_layers) fail("VELOX_FP32_LAYERS: layer " + tok + " does not exist");
                    else fp32_flags[l] = true;
                }
            }
            std::cerr << "[diag] FP32 weights for layers: " << spec << std::endl;
        } catch (const std::exception&) {
            fail("VELOX_FP32_LAYERS: could not parse '" + spec + "' (use 'all' or e.g. '2,3,21')");
        }
    }

    copy_tensor(weights_.final_norm, "model.norm.weight", hidden);
''', "final_norm copy line")

cpp = replace_once(cpp,
    '        copy_tensor(lw.post_attn_layernorm_w, prefix + ".post_attention_layernorm.weight", hidden);\n',
    '''        copy_tensor(lw.post_attn_layernorm_w, prefix + ".post_attention_layernorm.weight", hidden);

        if (fp32_flags[i]) {
            copy_tensor(lw.q_f32,    prefix + ".self_attn.q_proj.weight", q_dim, hidden);
            copy_tensor(lw.k_f32,    prefix + ".self_attn.k_proj.weight", kv_dim, hidden);
            copy_tensor(lw.v_f32,    prefix + ".self_attn.v_proj.weight", kv_dim, hidden);
            copy_tensor(lw.o_f32,    prefix + ".self_attn.o_proj.weight", hidden, q_dim);
            copy_tensor(lw.gate_f32, prefix + ".mlp.gate_proj.weight", inter, hidden);
            copy_tensor(lw.up_f32,   prefix + ".mlp.up_proj.weight", inter, hidden);
            copy_tensor(lw.down_f32, prefix + ".mlp.down_proj.weight", hidden, inter);
        }
''', "post_attention_layernorm copy line")

cpp = replace_once(cpp,
    "        LayerWeights& lw = weights_.layers[layer];\n",
    '''        LayerWeights& lw = weights_.layers[layer];
        // Projection: FP32 weights if this layer was selected by VELOX_FP32_LAYERS, otherwise INT8
        auto proj = [&](float* out, const float* in, const Int8Tensor& wq,
                        const std::vector<float>& wf, int M_, int N_, int K_) {
            if (!wf.empty()) ops::matmul(out, in, wf.data(), M_, N_, K_);
            else             ops::matmul_w8a8(out, in, wq, M_, N_, K_);
        };
''', "LayerWeights& lw in forward_batch")

calls = [
 ("ops::matmul_w8a8(q_buf_.data(), norm_out_.data(), lw.q_proj, total_positions, q_dim, hidden_size);",
  "proj(q_buf_.data(), norm_out_.data(), lw.q_proj, lw.q_f32, total_positions, q_dim, hidden_size);"),
 ("ops::matmul_w8a8(k_buf_.data(), norm_out_.data(), lw.k_proj, total_positions, kv_dim, hidden_size);",
  "proj(k_buf_.data(), norm_out_.data(), lw.k_proj, lw.k_f32, total_positions, kv_dim, hidden_size);"),
 ("ops::matmul_w8a8(v_buf_.data(), norm_out_.data(), lw.v_proj, total_positions, kv_dim, hidden_size);",
  "proj(v_buf_.data(), norm_out_.data(), lw.v_proj, lw.v_f32, total_positions, kv_dim, hidden_size);"),
 ("ops::matmul_w8a8(hidden_.data(), attn_out_.data(), lw.o_proj, total_positions, hidden_size, q_dim);",
  "proj(hidden_.data(), attn_out_.data(), lw.o_proj, lw.o_f32, total_positions, hidden_size, q_dim);"),
 ("ops::matmul_w8a8(ffn_gate_.data(), norm_out_.data(), lw.gate_proj, total_positions, intermediate_size, hidden_size);",
  "proj(ffn_gate_.data(), norm_out_.data(), lw.gate_proj, lw.gate_f32, total_positions, intermediate_size, hidden_size);"),
 ("ops::matmul_w8a8(ffn_up_.data(), norm_out_.data(), lw.up_proj, total_positions, intermediate_size, hidden_size);",
  "proj(ffn_up_.data(), norm_out_.data(), lw.up_proj, lw.up_f32, total_positions, intermediate_size, hidden_size);"),
 ("ops::matmul_w8a8(hidden_.data(), ffn_gate_.data(), lw.down_proj, total_positions, hidden_size, intermediate_size);",
  "proj(hidden_.data(), ffn_gate_.data(), lw.down_proj, lw.down_f32, total_positions, hidden_size, intermediate_size);"),
]
for old, new in calls:
    cpp = replace_once(cpp, old, new, old[:48] + "...")

cpp = replace_once(cpp,
    "    ops::matmul_w8a8(logits, norm_out_.data(), weights_.lm_head, batch_size, config_.vocab_size, config_.hidden_size);\n",
    '''    // Diagnostic: VELOX_FP32_LMHEAD=1 uses the FP32 embedding table directly (tied models only)
    static const bool fp32_lm_head = [this] {
        bool on = std::getenv("VELOX_FP32_LMHEAD") != nullptr;
        if (on && !config_.tie_word_embeddings)
            std::cerr << "[diag] VELOX_FP32_LMHEAD ignored: this model does not tie its embeddings" << std::endl;
        else if (on)
            std::cerr << "[diag] FP32 LM head (tied embeddings)" << std::endl;
        return on && config_.tie_word_embeddings;
    }();
    if (fp32_lm_head) ops::matmul(logits, norm_out_.data(), weights_.embed_tokens.data(), batch_size, config_.vocab_size, config_.hidden_size);
    else              ops::matmul_w8a8(logits, norm_out_.data(), weights_.lm_head, batch_size, config_.vocab_size, config_.hidden_size);
''', "lm_head matmul call")

open(hpp_path, "w").write(hpp)
open(cpp_path, "w").write(cpp)
print("applied: include/model.hpp and src/model.cpp updated (rebuild, then set VELOX_FP32_LAYERS)")
