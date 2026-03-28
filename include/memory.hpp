#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

template<typename T>
class Memory {
public:
    explicit Memory(std::size_t size)
        : buffer_(size) {}

    void reset() {
        offset_ = 0;
    }

    T* alloc(std::size_t size) {
        if (offset_ + size > buffer_.size()) {
            throw std::runtime_error("Out of memory");
        }

        T* ptr = buffer_.data() + offset_;
        offset_ += size;
        return ptr;
    }

    std::size_t used() const {
        return offset_;
    }

    std::size_t capacity() const {
        return buffer_.size();
    }

private:
    std::vector<T> buffer_;
    std::size_t offset_ = 0;
};
