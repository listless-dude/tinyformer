#include <algorithm>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

template<typename T>
class Tensor
{
public:
    std::vector<int> shape;
    std::vector<int> stride;
    std::unique_ptr<T[]> data;
    int size;
    
    Tensor(const std::vector<int>& shape);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    T& operator[](int idx);
    const T& operator[](int idx) const;

    Tensor add(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    void fill(T value);
    void print() const;

private:
    static int compute_size(const std::vector<int>& shape);
    void compute_strides();
};

// Constructor
template<typename T>
Tensor<T>::Tensor(const std::vector<int>& shape)
    : shape(shape), size(compute_size(shape))
{
    data = std::make_unique<T[]>(size);
    compute_strides();
}

// Copy Constructor
template<typename T>
Tensor<T>::Tensor(const Tensor& other) 
    : shape(other.shape), stride(other.stride), size(other.size)
{
    data = std::make_unique<T[]>(size);
    std::copy(other.data.get(), other.data.get()+size, data.get());
}

// Move Constructor
template<typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept
    : shape(std::move(other.shape)),
    stride(std::move(other.stride)),
    size(other.size),
    data(std::move(other.data))
{
    other.size = 0;
}

// Copy assignment
template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other)
{
    // same tensor
    if (this == &other)
        return *this;
    
    shape = other.shape;
    stride = other.stride;
    size = other.size;

    data = std::make_unique<T[]>(size);
    std::copy(other.data.get(), other.data.get()+size, data.get());

    return *this;
}

// Move assignment
template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept
{
    if (this == &other)
        return *this;

    shape = std::move(other.shape);
    stride = std::move(other.stride);
    size = other.size;
    
    data = std::move(other.data);
    return *this;
}

// Access
template<typename T>
T& Tensor<T>::operator[](int idx) 
{
    return data[idx];
}

template<typename T>
const T& Tensor<T>::operator[](int idx) const
{
    return data[idx];
}

template<typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& other) const {
    Tensor<T> result(shape);

    const T* a = data.get();
    const T* b = other.data.get();
    T* r = result.data.get();

    for (int i = 0; i < size; i++)
    {
        r[i] = a[i] + b[i];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
    Tensor<T> result(shape);

    (void)other;
    return result;
}

template<typename T>
void Tensor<T>::fill(T value)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = value;
    }
}

template<typename T>
void Tensor<T>::print() const
{
    for (int i = 0; i < size; i++)
    {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";
}

template<typename T>
int Tensor<T>::compute_size(const std::vector<int>& shape)
{
    int total = 1;
    for (int dim : shape)
    {
        total *= dim;
    }
    return total;
}

template<typename T>
void Tensor<T>::compute_strides()
{
    stride.resize(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}
