#include "tensor.hpp"

int main()
{
    Tensor<int> A({2, 3});
    Tensor<int> B({3, 2});

    A.fill(1);
    B.fill(2);

    A.print();
    B.print();

    Tensor<int> C = A.add(B);

    C.print();
    return 0;
}
