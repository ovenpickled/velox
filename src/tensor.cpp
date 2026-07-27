#include "tensor.hpp"
#include <numeric>

Tensor::Tensor() : data_(nullptr), owns_data_(false) {}

Tensor::Tensor(std::vector<int> shape) : shape_(std::move(shape)), owns_data_(true) {
    int total_size = size();
    if (total_size > 0) {
        owned_data_.resize(total_size, 0.0f);
        data_ = owned_data_.data();
    }
}

Tensor::Tensor(float* data, std::vector<int> shape) : data_(data), shape_(std::move(shape)), owns_data_(false) {}

Tensor::Tensor(const Tensor& other) : shape_(other.shape_), owns_data_(other.owns_data_) {
    if (owns_data_) {
        owned_data_ = other.owned_data_;
        data_ = owned_data_.data();
    } else {
        data_ = other.data_;
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        owns_data_ = other.owns_data_;
        if (owns_data_) {
            owned_data_ = other.owned_data_;
            data_ = owned_data_.data();
        } else {
            data_ = other.data_;
        }
    }
    return *this;
}

float* Tensor::data() { return data_; }
const float* Tensor::data() const { return data_; }
const std::vector<int>& Tensor::shape() const { return shape_; }
int Tensor::shape(int dim) const { return shape_[dim]; }
int Tensor::ndim() const { return shape_.size(); }

int Tensor::size() const {
    if (shape_.empty()) return 0;
    int total = 1;
    for (int s : shape_) total *= s;
    return total;
}

int Tensor::stride(int dim) const {
    int s = 1;
    for (int i = dim + 1; i < ndim(); ++i) {
        s *= shape_[i];
    }
    return s;
}

bool Tensor::empty() const { return data_ == nullptr || size() == 0; }

float& Tensor::operator()(int i) { return data_[i]; }
const float& Tensor::operator()(int i) const { return data_[i]; }
float& Tensor::operator()(int i, int j) { return data_[i * shape_[1] + j]; }
const float& Tensor::operator()(int i, int j) const { return data_[i * shape_[1] + j]; }

Tensor Tensor::view(std::vector<int> new_shape) const {
    int new_size = 1;
    for (int s : new_shape) new_size *= s;
    assert(new_size == size());
    return Tensor(data_, new_shape);
}

Tensor Tensor::row(int i) const {
    assert(ndim() == 2);
    return Tensor(data_ + i * shape_[1], {shape_[1]});
}

void Tensor::copy_from(const Tensor& other) {
    assert(size() == other.size());
    std::memcpy(data_, other.data_, size() * sizeof(float));
}

void Tensor::fill(float val) {
    int n = size();
    for (int i = 0; i < n; ++i) {
        data_[i] = val;
    }
}
