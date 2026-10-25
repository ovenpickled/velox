#include "safetensors.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include "json.hpp"

using json = nlohmann::json;

SafetensorsFile::SafetensorsFile() {}

SafetensorsFile::~SafetensorsFile() {
    close();
}

bool SafetensorsFile::open(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;

    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        close();
        return false;
    }
    file_size_ = sb.st_size;

    mapped_data_ = mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        mapped_data_ = nullptr;
        close();
        return false;
    }

    if (file_size_ < 8) { close(); return false; }
    uint64_t header_length;
    std::memcpy(&header_length, mapped_data_, 8);
    if (header_length > file_size_ - 8) {
        std::cerr << "Corrupt safetensors header (length " << header_length << " exceeds file size)" << std::endl;
        close();
        return false;
    }
    header_size_ = header_length;

    const char* json_start = static_cast<const char*>(mapped_data_) + 8;
    std::string header_str(json_start, header_length);

    json metadata;
    try {
        metadata = json::parse(header_str);
    } catch (...) {
        close();
        return false;
    }

    data_start_ = static_cast<const uint8_t*>(mapped_data_) + 8 + header_length;

    const size_t data_bytes = file_size_ - 8 - header_length;
    try {
        for (auto it = metadata.begin(); it != metadata.end(); ++it) {
            if (it.key() == "__metadata__") continue;

            TensorInfo info;
            info.name = it.key();
            info.dtype = it.value()["dtype"];
            for (const auto& dim : it.value()["shape"]) {
                info.shape.push_back(dim.get<int>());
            }
            size_t begin = it.value()["data_offsets"][0].get<size_t>();
            size_t end = it.value()["data_offsets"][1].get<size_t>();
            if (end < begin || end > data_bytes) {
                std::cerr << "Tensor " << info.name << " points outside the file (truncated download?)" << std::endl;
                close();
                return false;
            }
            info.data_offset = begin;
            info.data_size = end - begin;
            tensors_[info.name] = info;
        }
    } catch (const std::exception& e) {
        std::cerr << "Malformed safetensors header: " << e.what() << std::endl;
        close();
        return false;
    }

    return true;
}

void SafetensorsFile::close() {
    if (mapped_data_ && mapped_data_ != MAP_FAILED) {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
    tensors_.clear();
    converted_tensors_.clear();
}

void SafetensorsFile::clear() {
    converted_tensors_.clear();
}

bool SafetensorsFile::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

TensorInfo SafetensorsFile::get_tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) return it->second;
    return TensorInfo();
}

std::vector<std::string> SafetensorsFile::tensor_names() const {
    std::vector<std::string> names;
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

float SafetensorsFile::bf16_to_float(uint16_t val) const {
    float result;
    uint32_t f32 = static_cast<uint32_t>(val) << 16;
    std::memcpy(&result, &f32, 4);
    return result;
}

float SafetensorsFile::f16_to_float(uint16_t h) const {
    uint32_t sign = static_cast<uint32_t>(h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;                                   // +-0
        } else {                                              // subnormal half -> normal float
            int shift = 0;
            while (!(mant & 0x400u)) { mant <<= 1; shift++; }
            mant &= 0x3FFu;
            f = (sign << 31) | (static_cast<uint32_t>(127 - 15 - shift + 1) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);      // inf / NaN
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

Tensor SafetensorsFile::get_tensor(const std::string& name) {
    if (!has_tensor(name)) return Tensor();

    const TensorInfo& info = tensors_[name];
    const uint8_t* raw_data = data_start_ + info.data_offset;

    if (info.dtype == "F32") {
        return Tensor(const_cast<float*>(reinterpret_cast<const float*>(raw_data)), info.shape);
    } else if (info.dtype == "BF16") {
        size_t num_elements = info.data_size / 2;
        std::vector<float> converted(num_elements);
        const uint16_t* bf16_data = reinterpret_cast<const uint16_t*>(raw_data);
        for (size_t i = 0; i < num_elements; ++i) {
            converted[i] = bf16_to_float(bf16_data[i]);
        }
        converted_tensors_.push_back(std::move(converted));
        return Tensor(converted_tensors_.back().data(), info.shape);
    } else if (info.dtype == "F16") {
        size_t num_elements = info.data_size / 2;
        std::vector<float> converted(num_elements);
        const uint16_t* f16_data = reinterpret_cast<const uint16_t*>(raw_data);
        for (size_t i = 0; i < num_elements; ++i) {
            converted[i] = f16_to_float(f16_data[i]);
        }
        converted_tensors_.push_back(std::move(converted));
        return Tensor(converted_tensors_.back().data(), info.shape);
    }

    std::cerr << "Unsupported dtype '" << info.dtype << "' for tensor " << name
              << " (supported: F32, BF16, F16)" << std::endl;
    return Tensor();
}
