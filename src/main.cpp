#include <iostream>

#include "executor.hpp"
#include "graph.hpp"
#include "memory.hpp"
#include "ops.hpp"

int main() {
    Memory<float> arena(64);

    Tensor<float> a({2, 2}, arena.alloc(4));
    Tensor<float> b({2, 2}, arena.alloc(4));
    Tensor<float> bias({2, 2}, arena.alloc(4));
    Tensor<float> mm_out({2, 2}, arena.alloc(4));
    Tensor<float> add_out({2, 2}, arena.alloc(4));
    Tensor<float> relu_out({2, 2}, arena.alloc(4));

    a.data[0] = 1.0f;  a.data[1] = 2.0f;
    a.data[2] = 3.0f;  a.data[3] = 4.0f;

    b.data[0] = 5.0f;  b.data[1] = 6.0f;
    b.data[2] = 7.0f;  b.data[3] = 8.0f;

    bias.data[0] = -20.0f; bias.data[1] = 1.0f;
    bias.data[2] = 2.0f;   bias.data[3] = -50.0f;

    MatMulOp<float> matmul(&a, &b, &mm_out);
    AddOp<float> add(&mm_out, &bias, &add_out);
    ReluOp<float> relu(&add_out, &relu_out);

    Graph<float> graph;
    graph.add_op(&matmul);
    graph.add_op(&add);
    graph.add_op(&relu);

    Executor<float> executor;
    executor.run(graph);

    std::cout << "MatMul + Add + ReLU output:\n";
    for (int r = 0; r < relu_out.rows(); ++r) {
        for (int c = 0; c < relu_out.cols(); ++c) {
            std::cout << relu_out.data[r * relu_out.cols() + c] << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}
