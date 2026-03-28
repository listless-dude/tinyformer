#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

enum OpType {
    NONE,
    MATMUL,
    ADD,
    RELU
};

template<typename T>
struct Tensor {
    std::vector<int> shape;
    T* data = nullptr;
    bool computed = false;

    Tensor() = default;

    Tensor(std::vector<int> dims, T* ptr = nullptr)
        : shape(std::move(dims)), data(ptr) {}

    int rows() const {
        if (shape.size() != 2) {
            throw std::runtime_error("Tensor is not rank-2");
        }
        return shape[0];
    }

    int cols() const {
        if (shape.size() != 2) {
            throw std::runtime_error("Tensor is not rank-2");
        }
        return shape[1];
    }

    std::size_t numel() const {
        if (shape.empty()) {
            return 0;
        }

        std::size_t total = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error("Tensor shape must be positive");
            }
            total *= static_cast<std::size_t>(dim);
        }
        return total;
    }

    bool has_data() const {
        return data != nullptr;
    }
};
