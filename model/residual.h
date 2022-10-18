//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_RESIDUAL_H
#define ATTENTION_TRANSFORMER_CPP_RESIDUAL_H

#include "layer.h"
#include "tensor.h"
#include "layer_norm.h"

namespace text_attention {
template<typename T>
class Residual : virtual public Layer<T> {
public:
    Residual() : fn(nullptr) {
    }

    explicit Residual(Layer<T> *fn) : fn(fn) {
    }

    uint64_t parameterCount() override {
        if (fn != nullptr) {
            return fn->parameterCount();
        }
        return 0;
    }

    void forward(const Tensor<T> &input, Tensor<T> &output, 
            const Tensor<bool> &mask, const Tensor<T> &memory) override {
        assert(fn != nullptr);
        fn->forward(input, output, mask, memory);

        /* Residual connection */
        for (int i = 0; i < output.size(); ++i) 
            output[i] += input[i];
    }


private:
    Layer<T> *fn;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_RESIDUAL_H
