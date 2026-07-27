#include "safetensors.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
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

    uint64_t header_length;
    std::memcpy(&header_length, mapped_data_, 8);
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

    for (auto it = metadata.begin(); it != metadata.end(); ++it) {
        if (it.key() == "__metadata__") continue;

        TensorInfo info;
        info.name = it.key();
        info.dtype = it.value()["dtype"];
        for (const auto& dim : it.value()["shape"]) {
            info.shape.push_back(dim.get<int>());
        }
        info.data_offset = it.value()["data_offsets"][0].get<size_t>();
        info.data_size = it.value()["data_offsets"][1].get<size_t>() - info.data_offset;
        tensors_[info.name] = info;
    }

    return true;
}

void SafetensorsFile::close() {
    if (mapped_data_) {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
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
    }

    return Tensor();
}
