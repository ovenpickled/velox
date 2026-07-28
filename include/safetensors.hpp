#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.hpp"
#include <cstdint>

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int> shape;
    size_t data_offset;
    size_t data_size;
};

class SafetensorsFile {
public:
    SafetensorsFile();
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    bool open(const std::string& path);
    void close();
    void clear();

    bool has_tensor(const std::string& name) const;
    TensorInfo get_tensor_info(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

    Tensor get_tensor(const std::string& name);

private:
    int fd_ = -1;
    void* mapped_data_ = nullptr;
    size_t file_size_ = 0;
    size_t header_size_ = 0;
    const uint8_t* data_start_ = nullptr;

    std::unordered_map<std::string, TensorInfo> tensors_;
    std::vector<std::vector<float>> converted_tensors_;

    float bf16_to_float(uint16_t val) const;
};
