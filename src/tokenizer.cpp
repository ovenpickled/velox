#include "tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "json.hpp"

using json = nlohmann::json;

// --- UTF-8 Utilities ---

std::string Tokenizer::unicode_to_utf8(int cp) {
    std::string result;
    if (cp < 0x80) {
        result += static_cast<char>(cp);
    } else if (cp < 0x800) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

std::vector<std::string> Tokenizer::utf8_split(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = 1;
        unsigned char c = static_cast<unsigned char>(s[i]);
        if ((c & 0xF8) == 0xF0) len = 4;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xE0) == 0xC0) len = 2;
        if (i + len <= s.size()) {
            chars.push_back(s.substr(i, len));
        }
        i += len;
    }
    return chars;
}

// --- Byte Encoder (GPT-2 byte-to-unicode mapping) ---

void Tokenizer::build_byte_encoder() {
    std::vector<int> bs;
    for (int i = 0x21; i <= 0x7E; i++) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);

    std::vector<int> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); i++) {
        std::string utf8_char = unicode_to_utf8(cs[i]);
        byte_encoder_[static_cast<uint8_t>(bs[i])] = utf8_char;
        byte_decoder_[utf8_char] = static_cast<uint8_t>(bs[i]);
    }
}

// --- Tokenizer Loading ---

bool Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path);
    if (!f.is_open()) return false;
    json data;
    f >> data;

    build_byte_encoder();

    if (data.contains("model") && data["model"].contains("vocab")) {
        for (auto& [key, value] : data["model"]["vocab"].items()) {
            vocab_[key] = value;
            id_to_token_[value] = key;
        }
    }
    vocab_size_ = vocab_.size();

    if (data.contains("model") && data["model"].contains("merges")) {
        int rank = 0;
        for (const auto& merge : data["model"]["merges"]) {
            std::string merge_str = merge;
            size_t space = merge_str.find(' ');
            if (space != std::string::npos) {
                std::string a = merge_str.substr(0, space);
                std::string b = merge_str.substr(space + 1);
                merges_.push_back({a, b});
                merge_ranks_[{a, b}] = rank++;
            }
        }
    }

    if (data.contains("added_tokens")) {
        for (const auto& token_obj : data["added_tokens"]) {
            std::string content = token_obj["content"];
            int id = token_obj["id"];
            vocab_[content] = id;
            id_to_token_[id] = content;
            if (content.find("endoftext") != std::string::npos) {
                eos_id_ = id;
            }
            if (content.find("beginoftext") != std::string::npos) {
                bos_id_ = id;
            }
        }
    }

    if (eos_id_ == -1 && vocab_.count("<|endoftext|>")) {
        eos_id_ = vocab_["<|endoftext|>"];
    }
    if (eos_id_ == -1 && vocab_.count("<|im_end|>")) {
        eos_id_ = vocab_["<|im_end|>"];
    }
    if (bos_id_ == -1) {
        bos_id_ = eos_id_;
    }

    return true;
}

// --- Pre-tokenization ---

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> chunks;
    std::string current;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            if (!current.empty()) {
                chunks.push_back(current);
                current.clear();
            }
            current += c;
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        chunks.push_back(current);
    }
    return chunks;
}

// --- BPE ---

std::vector<std::string> Tokenizer::bpe(const std::string& token) const {
    std::vector<std::string> word = utf8_split(token);

    if (word.size() <= 1) return word;

    while (word.size() > 1) {
        int best_rank = -1;
        std::pair<std::string, std::string> best_pair;
        int best_idx = -1;

        for (size_t i = 0; i < word.size() - 1; ++i) {
            auto pair = std::make_pair(word[i], word[i + 1]);
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end()) {
                if (best_rank == -1 || it->second < best_rank) {
                    best_rank = it->second;
                    best_pair = pair;
                    best_idx = i;
                }
            }
        }

        if (best_rank == -1) break;

        std::string merged = best_pair.first + best_pair.second;
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ++i) {
            if (i < word.size() - 1 && word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                new_word.push_back(merged);
                i++;
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = new_word;
    }
    return word;
}

// --- Encode ---

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    std::vector<std::string> chunks = pre_tokenize(text);

    for (const auto& chunk : chunks) {
        std::string encoded;
        for (unsigned char c : chunk) {
            auto it = byte_encoder_.find(c);
            if (it != byte_encoder_.end()) {
                encoded += it->second;
            }
        }

        std::vector<std::string> bpe_tokens = bpe(encoded);
        for (const auto& t : bpe_tokens) {
            auto vit = vocab_.find(t);
            if (vit != vocab_.end()) {
                ids.push_back(vit->second);
            }
        }
    }
    return ids;
}

// --- Decode ---

std::string Tokenizer::decode(int token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it == id_to_token_.end()) return "";

    const std::string& token = it->second;
    std::string result;
    auto chars = utf8_split(token);
    for (const auto& ch : chars) {
        auto dit = byte_decoder_.find(ch);
        if (dit != byte_decoder_.end()) {
            result += static_cast<char>(dit->second);
        } else {
            result += ch;
        }
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string text;
    for (int id : token_ids) {
        text += decode(id);
    }
    return text;
}

int Tokenizer::vocab_size() const { return vocab_size_; }
int Tokenizer::bos_token_id() const { return bos_id_; }
int Tokenizer::eos_token_id() const { return eos_id_; }
