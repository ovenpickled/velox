#pragma once
#include <vector>
#include <cstring>
#include <cassert>
#include <cmath>
#include <string>
#include <cstdint>

class Tensor {
public:
    Tensor();
    explicit Tensor(std::vector<int> shape);
    Tensor(float* data, std::vector<int> shape);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    float* data();
    const float* data() const;
    const std::vector<int>& shape() const;
    int shape(int dim) const;
    int ndim() const;
    int size() const;
    int stride(int dim) const;
    bool empty() const;

    float& operator()(int i);
    const float& operator()(int i) const;
    float& operator()(int i, int j);
    const float& operator()(int i, int j) const;

    Tensor view(std::vector<int> new_shape) const;
    Tensor row(int i) const;

    void copy_from(const Tensor& other);
    void fill(float val);

private:
    std::vector<float> owned_data_;
    float* data_ = nullptr;
    std::vector<int> shape_;
    bool owns_data_ = false;
};

struct Int8Tensor {
    std::vector<int8_t> data;
    std::vector<float> scales; // one scale per row
    int M = 0; // number of rows (output channels)
    int K = 0; // number of cols (input channels)
};
