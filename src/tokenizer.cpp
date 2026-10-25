#include "tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include "json.hpp"
#include "unicode_tables.hpp"

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
    try {
        f >> data;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse " << tokenizer_json_path << ": " << e.what() << std::endl;
        return false;
    }

    build_byte_encoder();

    if (data.contains("model") && data["model"].contains("vocab")) {
        for (auto& [key, value] : data["model"]["vocab"].items()) {
            vocab_[key] = value;
            id_to_token_[value] = key;
        }
    }
    if (vocab_.empty()) {
        std::cerr << "Tokenizer has no vocabulary: " << tokenizer_json_path << std::endl;
        return false;
    }

    if (data.contains("model") && data["model"].contains("merges")) {
        int rank = 0;
        for (const auto& merge : data["model"]["merges"]) {
            std::string a, b;
            if (merge.is_string()) {                       // "a b"
                std::string merge_str = merge;
                size_t space = merge_str.find(' ');
                if (space == std::string::npos) continue;
                a = merge_str.substr(0, space);
                b = merge_str.substr(space + 1);
            } else if (merge.is_array() && merge.size() == 2) {   // ["a", "b"] (newer tokenizer.json)
                a = merge[0].get<std::string>();
                b = merge[1].get<std::string>();
            } else {
                continue;
            }
            merges_.push_back({a, b});
            merge_ranks_[{a, b}] = rank++;
        }
    }

    if (data.contains("added_tokens")) {
        for (const auto& token_obj : data["added_tokens"]) {
            std::string content = token_obj["content"];
            int id = token_obj["id"];
            vocab_[content] = id;
            id_to_token_[id] = content;
            if (!content.empty()) added_tokens_.push_back({content, id});
        }
    }
    vocab_size_ = static_cast<int>(vocab_.size());

    // Longest added token first, so "<|im_start|>" wins over any shorter token sharing a prefix
    std::sort(added_tokens_.begin(), added_tokens_.end(),
              [](const auto& x, const auto& y) { return x.first.size() > y.first.size(); });
    for (const auto& at : added_tokens_) added_first_byte_[static_cast<unsigned char>(at.first[0])] = true;

    // End-of-sequence tokens: exact names (no substring guessing)
    auto find_id = [&](const char* name) { auto it = vocab_.find(name); return it == vocab_.end() ? -1 : it->second; };
    int endoftext = find_id("<|endoftext|>");
    int im_end = find_id("<|im_end|>");
    eos_id_ = (endoftext != -1) ? endoftext : im_end;
    for (int id : {endoftext, im_end}) {
        if (id != -1 && std::find(eos_ids_.begin(), eos_ids_.end(), id) == eos_ids_.end()) eos_ids_.push_back(id);
    }
    int bos = find_id("<|beginoftext|>");
    bos_id_ = (bos != -1) ? bos : eos_id_;

    return true;
}

// --- Pre-tokenization ---
//
// Hand-written equivalent of the Qwen2 split regex (applied to UTF-8 text):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
// Alternatives are tried in this order at every position; the first that matches wins.

static bool in_ranges(const uint32_t (*ranges)[2], size_t count, uint32_t cp) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid][0]) hi = mid;
        else if (cp > ranges[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

static bool is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    return in_ranges(unicode_tables::kLetterRanges, unicode_tables::kLetterCount, cp);
}

static bool is_number(uint32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(unicode_tables::kNumberRanges, unicode_tables::kNumberCount, cp);
}

// Unicode White_Space property
static bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

static bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// Case folding for the contraction alternatives. U+017F (long s) folds to 's' under Unicode
// case-insensitive matching, so "'\u017F" is the contraction "'s".
static uint32_t ascii_lower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp == 0x17F) return 's';
    return cp;
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    // Decode UTF-8 into code points, remembering each one's byte offset so chunks keep the exact input bytes.
    // Invalid bytes become U+FFFD (not a letter, number or space) and advance by one byte.
    std::vector<uint32_t> cps;
    std::vector<size_t> pos;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int len = 1;
        uint32_t cp = 0xFFFD;
        if (c < 0x80) { cp = c; }
        else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        if (len > 1) {
            bool ok = i + len <= text.size();
            for (int k = 1; ok && k < len; k++) {
                unsigned char cc = static_cast<unsigned char>(text[i + k]);
                if ((cc & 0xC0) != 0x80) ok = false;
                else cp = (cp << 6) | (cc & 0x3F);
            }
            if (!ok) { len = 1; cp = 0xFFFD; }
        }
        cps.push_back(cp);
        pos.push_back(i);
        i += len;
    }
    const size_t n = cps.size();
    pos.push_back(text.size());

    auto L = [&](size_t k) { return k < n && is_letter(cps[k]); };
    auto N = [&](size_t k) { return k < n && is_number(cps[k]); };
    auto other = [&](size_t k) { return k < n && !is_space(cps[k]) && !is_letter(cps[k]) && !is_number(cps[k]); };

    std::vector<std::string> chunks;
    size_t i = 0;
    while (i < n) {
        size_t end = i;  // exclusive end of the chunk starting at i

        // 1. contractions (case-insensitive)
        bool matched = false;
        if (cps[i] == '\'' && i + 1 < n) {
            uint32_t c1 = ascii_lower(cps[i + 1]);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { end = i + 2; matched = true; }
            else if (i + 2 < n) {
                uint32_t c2 = ascii_lower(cps[i + 2]);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    end = i + 3; matched = true;
                }
            }
        }

        // 2. one optional non-letter/number/newline character, then letters
        if (!matched) {
            if (L(i)) {
                end = i; while (L(end)) end++;
                matched = true;
            } else if (i + 1 < n && !is_newline(cps[i]) && !N(i) && L(i + 1)) {
                end = i + 1; while (L(end)) end++;
                matched = true;
            }
        }

        // 3. a single digit
        if (!matched && N(i)) { end = i + 1; matched = true; }

        // 4. optional space, then punctuation/symbols, then trailing newlines
        if (!matched) {
            size_t j = i;
            bool ok = false;
            if (cps[i] == ' ' && other(i + 1)) { j = i + 1; ok = true; }
            else if (other(i)) { j = i; ok = true; }
            if (ok) {
                while (other(j)) j++;
                while (j < n && is_newline(cps[j])) j++;
                end = j; matched = true;
            }
        }

        if (!matched) {
            // cps[i] is whitespace here. Find the whitespace run [i, run_end).
            size_t run_end = i;
            while (run_end < n && is_space(cps[run_end])) run_end++;

            // 5. whitespace ending in a newline: up to and including the last newline in the run
            size_t last_nl = std::string::npos;
            for (size_t k = i; k < run_end; k++) if (is_newline(cps[k])) last_nl = k;
            if (last_nl != std::string::npos) {
                end = last_nl + 1; matched = true;
            }
            // 6. whitespace not followed by a non-space: leave the last space to attach to the next word
            else if (run_end == n) {
                end = run_end; matched = true;
            } else if (run_end - i >= 2) {
                end = run_end - 1; matched = true;
            }
            // 7. otherwise the whole (single) whitespace run
            else {
                end = run_end; matched = true;
            }
        }

        chunks.push_back(text.substr(pos[i], pos[end] - pos[i]));
        i = end;
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

        for (size_t i = 0; i < word.size() - 1; ++i) {
            auto pair = std::make_pair(word[i], word[i + 1]);
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end()) {
                if (best_rank == -1 || it->second < best_rank) {
                    best_rank = it->second;
                    best_pair = pair;
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

void Tokenizer::encode_segment(const std::string& text, std::vector<int>& ids) const {
    for (const auto& chunk : pre_tokenize(text)) {
        std::string encoded;
        for (unsigned char c : chunk) {
            auto it = byte_encoder_.find(c);
            if (it != byte_encoder_.end()) {
                encoded += it->second;
            }
        }

        for (const auto& t : bpe(encoded)) {
            auto vit = vocab_.find(t);
            if (vit != vocab_.end()) {
                ids.push_back(vit->second);
            } else {
                static bool warned = false;
                if (!warned) {
                    std::cerr << "Warning: tokenizer produced a piece that is not in the vocabulary; dropping it" << std::endl;
                    warned = true;
                }
            }
        }
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    size_t seg_start = 0, i = 0;
    while (i < text.size()) {
        if (added_first_byte_[static_cast<unsigned char>(text[i])]) {
            const std::pair<std::string, int>* hit = nullptr;
            for (const auto& at : added_tokens_) {   // longest first
                if (at.first.size() <= text.size() - i && text.compare(i, at.first.size(), at.first) == 0) {
                    hit = &at;
                    break;
                }
            }
            if (hit) {
                if (i > seg_start) encode_segment(text.substr(seg_start, i - seg_start), ids);
                ids.push_back(hit->second);
                i += hit->first.size();
                seg_start = i;
                continue;
            }
        }
        i++;
    }
    if (seg_start < text.size()) encode_segment(text.substr(seg_start), ids);
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

bool Tokenizer::is_eos(int token_id) const {
    return std::find(eos_ids_.begin(), eos_ids_.end(), token_id) != eos_ids_.end();
}
