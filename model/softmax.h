//
// Created by dianh on 2021/04/21.
//

#ifndef ATTENTION_TRANSFORMER_CPP_SOFTMAX_H
#define ATTENTION_TRANSFORMER_CPP_SOFTMAX_H

#include "layer.h"
#include "tensor.h"

namespace text_attention {
template<typename T>
class SoftMax : virtual public Layer<T> {
public:
    void forward(const Tensor <T> &input, Tensor <T> &output) override {
        int dim = input.shape[input.shape.size()-1];
        output.clear();
        output.reshape(input.shape);
        for (int i = 0; i < input.size(); i += dim) {
            T sum = 0;
            for (int j = 0; j < dim; ++j) {
                sum += exp(input[i + j]);
            }
            for (int j = 0; j < dim; ++j) {
                output[i+j] = (exp(input[i + j]) / sum);
            }
        }
    }

    uint64_t parameterCount() override {
        return 0;
    }
};
}
#endif //ATTENTION_TRANSFORMER_CPP_SOFTMAX_H
