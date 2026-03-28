#pragma once

#include "graph.hpp"
#include "ops.hpp"

template<typename T>
class Executor {
public:
    void run(const Graph<T>& graph) {
        for (Op<T>* op : graph.ops()) {
            op->compute();
        }
    }
};
