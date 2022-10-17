//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_PRE_NORM_H
#define ATTENTION_TRANSFORMER_CPP_PRE_NORM_H

#include "layer.h"
#include "layer_norm.h"
#include "functions.h"

namespace text_attention {
    template<typename T>
    class PreNorm : virtual public Layer<T> {
    public:
        PreNorm(Layer <T> *fn, int dim, std::string str_key_layer) : fn(fn), dim(dim) {
            layerNorm = new LayerNorm<T>(dim, text_attention::param_map[str_key_layer+"norm.a_2"].pvals, text_attention::param_map[str_key_layer+"norm.b_2"].pvals);
        }
    
        long long parameterCount() {
            long long ret = 0;
            if (fn != nullptr) {
                ret += fn->parameterCount();
            }
            ret += layerNorm->parameterCount();
            return ret;
        }

        void forward(const Tensor <T> &input, Tensor <T> &output, Tensor<T> &mask, Tensor<T> &memory) {
            Tensor<T> tmp{};
            fn->forward(input, tmp, mask, memory);
            layerNorm->forward(tmp, output);

        }

    private:
        Layer <T> *fn = nullptr;
        LayerNorm <T> *layerNorm = nullptr;
        int dim;
    };
}
#endif //ATTENTION_TRANSFORMER_CPP_PRE_NORM_H
