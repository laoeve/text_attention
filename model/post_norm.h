//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_POST_NORM_H
#define ATTENTION_TRANSFORMER_CPP_POST_NORM_H

#include "bits/stdc++.h"
#include "top_model.h"
#include "layer.h"
#include "layer_norm.h"
#include "functions.h"

using namespace std;

namespace text_attention {
template<typename T>
class PostNorm : virtual public Layer<T> {
public:
    PostNorm(TopModel<T>* master, Layer<T>* fn, 
            int dim_model, const string prefix_str, 
            const string LN_gamma_str, const string LN_beta_str)
    : Layer<T>(master), fn(fn)
    {
        std::cout << ">>>> Init layer normalization sublayer - " << std::endl;

        std::vector<T>* gamma = 
            new vector<T>(param_map[prefix_str+"."+LN_gamma_str].pvals);
        std::vector<T>* beta = 
            new vector<T>(param_map[prefix_str+"."+LN_beta_str].pvals);
        
        layerNorm = new LayerNorm<T>(prefix_str, dim_model, *gamma, *beta);
        layerNorm->print_params( );
    }

    uint64_t parameterCount() override {
        uint64_t ret = 0;
        if (fn != nullptr) {
            ret += fn->parameterCount();
        }
        ret += layerNorm->parameterCount();
        return ret;
    }

    void forward(Tensor <T> &output, const Tensor <T> &input, 
            const Tensor<bool> &mask, const Tensor<T> &memory) override {
        Tensor<T> fn2ln{};
        fn->forward(fn2ln, input, mask, memory);
        layerNorm->forward(output, fn2ln);
    }

private:
    Layer <T> *fn = nullptr; // can be either multihead attention or feedforward
    LayerNorm <T> *layerNorm = nullptr;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_POST_NORM_H
