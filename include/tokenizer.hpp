#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>

class Tokenizer {
public:
    bool load(const std::string& tokenizer_json_path);

    // Encodes text. Added/special tokens that appear literally in the text (e.g. "<|im_start|>")
    // are mapped to their single token ids, like Hugging Face does by default.
    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& token_ids) const;

    int vocab_size() const;
    int bos_token_id() const;
    int eos_token_id() const;
    // True for any token that should stop generation (<|endoftext|> and <|im_end|> when present).
    bool is_eos(int token_id) const;

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;

    int vocab_size_ = 0;
    int bos_id_ = -1;
    int eos_id_ = -1;
    std::vector<int> eos_ids_;

    // Added tokens (special tokens etc.), longest first, matched literally inside the input text
    std::vector<std::pair<std::string, int>> added_tokens_;
    bool added_first_byte_[256] = {false};

    // GPT-2 byte-level encoding
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    void build_byte_encoder();
    static std::string unicode_to_utf8(int codepoint);
    static std::vector<std::string> utf8_split(const std::string& s);

    std::vector<std::string> bpe(const std::string& token) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    void encode_segment(const std::string& text, std::vector<int>& ids) const;
};
