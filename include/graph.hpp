#pragma once

#include <vector>

template<typename T>
class Op;

template<typename T>
class Graph {
public:
    void add_op(Op<T>* op) {
        ops_.push_back(op);
    }

    const std::vector<Op<T>*>& ops() const {
        return ops_;
    }

private:
    std::vector<Op<T>*> ops_;
};
