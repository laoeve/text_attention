//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_PRE_NORM_H
#define ATTENTION_TRANSFORMER_CPP_PRE_NORM_H

#include "top_model.h"
#include "layer.h"
#include "layer_norm.h"
#include "functions.h"

using namespace std;

namespace text_attention {
template<typename T>
class PreNorm : virtual public Layer<T> {
public:
    PreNorm(TopModel<T>* master, Layer<T>* fn, 
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

    void forward(const Tensor <T> &input, Tensor <T> &output, 
            Tensor<T> &mask, Tensor<T> &memory) override {
        Tensor<T> tmp{};
        layerNorm->forward(input, tmp);
        fn->forward(tmp, output, mask, memory);
    }

private:
    Layer <T> *fn = nullptr;
    LayerNorm <T> *layerNorm = nullptr;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_PRE_NORM_H
