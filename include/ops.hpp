#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "tensor.hpp"

template<typename T>
class Op {
public:
    virtual void compute() = 0;
    virtual ~Op() = default;
};

namespace detail {
template<typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<typename T>
void require_data(const Tensor<T>* tensor, const std::string& name) {
    if (tensor == nullptr) {
        throw std::runtime_error(name + " tensor pointer is null");
    }
    if (!tensor->has_data()) {
        throw std::runtime_error(name + " tensor has no allocated data");
    }
}

template<typename T>
void require_same_shape(const Tensor<T>* a, const Tensor<T>* b, const Tensor<T>* out, const std::string& op_name) {
    if (a->shape != b->shape || a->shape != out->shape) {
        throw std::runtime_error(op_name + " expects all tensors to have identical shapes");
    }
}
}  // namespace detail

template<typename T>
class MatMulOp : public Op<T> {
public:
    MatMulOp(Tensor<T>* a, Tensor<T>* b, Tensor<T>* c)
        : A(a), B(b), C(c) {}

    void compute() override {
        detail::require_data(A, "A");
        detail::require_data(B, "B");
        detail::require_data(C, "C");

        if (A->cols() != B->rows()) {
            throw std::runtime_error("MatMul dimension mismatch");
        }
        if (C->rows() != A->rows() || C->cols() != B->cols()) {
            throw std::runtime_error("MatMul output shape mismatch");
        }

        Eigen::Map<detail::RowMajorMatrix<T>> matA(A->data, A->rows(), A->cols());
        Eigen::Map<detail::RowMajorMatrix<T>> matB(B->data, B->rows(), B->cols());
        Eigen::Map<detail::RowMajorMatrix<T>> matC(C->data, C->rows(), C->cols());

        matC.noalias() = matA * matB;
        C->computed = true;
    }

private:
    Tensor<T>* A;
    Tensor<T>* B;
    Tensor<T>* C;
};

template<typename T>
class AddOp : public Op<T> {
public:
    AddOp(Tensor<T>* a, Tensor<T>* b, Tensor<T>* c)
        : A(a), B(b), C(c) {}

    void compute() override {
        detail::require_data(A, "A");
        detail::require_data(B, "B");
        detail::require_data(C, "C");
        detail::require_same_shape(A, B, C, "Add");

        const std::size_t size = A->numel();
        for (std::size_t i = 0; i < size; ++i) {
            C->data[i] = A->data[i] + B->data[i];
        }
        C->computed = true;
    }

private:
    Tensor<T>* A;
    Tensor<T>* B;
    Tensor<T>* C;
};

template<typename T>
class ReluOp : public Op<T> {
public:
    ReluOp(Tensor<T>* a, Tensor<T>* c)
        : A(a), C(c) {}

    void compute() override {
        detail::require_data(A, "A");
        detail::require_data(C, "C");

        if (A->shape != C->shape) {
            throw std::runtime_error("ReLU expects input and output shapes to match");
        }

        const std::size_t size = A->numel();
        for (std::size_t i = 0; i < size; ++i) {
            C->data[i] = std::max(static_cast<T>(0), A->data[i]);
        }
        C->computed = true;
    }

private:
    Tensor<T>* A;
    Tensor<T>* C;
};
