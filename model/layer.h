//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_LAYER_H
#define ATTENTION_TRANSFORMER_CPP_LAYER_H

#include <bits/stdc++.h>
#include "tensor.h"

namespace text_attention {
    template<typename T>
    class Layer {
    public:
        virtual void forward(const Tensor<T> &input, Tensor<T> &output, Tensor<T> &mask, Tensor<T> &memory) {

        }

        virtual long long parameterCount() {
            return 0;
        }
    };

}
#endif //ATTENTION_TRANSFORMER_CPP_LAYER_H
