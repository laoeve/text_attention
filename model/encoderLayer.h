//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// based from swin_block.h

#ifndef ATTENTION_TRANSFORMER_CPP_ENCODER_LAYER_H
#define ATTENTION_TRANSFORMER_CPP_ENCODER_LAYER_H

#include "top_model.h"
#include "layer.h"
#include "residual.h"
#include "pre_norm.h"
#include "multiheadattention.h"
#include "feed_forward.h"

using namespace std;

namespace text_attention {
template<typename T>
class EncoderLayer : virtual public Layer<T> {
public:
    EncoderLayer(TopModel<T>* master,
            int dim_model, int num_heads, int dim_ff, const string prefix_enc, 
            const string prefix_layer, int id, const string weight_str, 
            const string bias_str, const string LN_gamma_str, 
            const string LN_beta_str, const string sa_query_str, 
            const string sa_key_str, const string sa_value_str, 
            const string sa_out_str, const string ff_hidden_str, 
            const string ff_out_str, const string LN_mh_str, const string LN_ff_str) 
    : Layer<T>(master)
    {
        string prefix_str = prefix_enc+"."+prefix_layer+"."+to_string(id);
        std::cout << "Init encoder - " << prefix_str << std::endl;
        
        /* Init attention layer */
        multiheadAttention = new MultiheadAttention<T>(master, 
                dim_model, num_heads, prefix_str, weight_str, bias_str, 
                sa_query_str, sa_key_str, sa_value_str, sa_out_str);

        /* Init feedforward layer (MLP) */
        feedForward = new FeedForward<T>(master, dim_model, dim_ff,
                prefix_str, weight_str, bias_str, ff_hidden_str, ff_out_str);

        /* Init layer normalizations */
        preNorm_mh = new PreNorm<T>(master, multiheadAttention, dim_model,
                prefix_str+"."+LN_mh_str, LN_gamma_str, LN_beta_str);
        residual_mh = new Residual<T>(preNorm_mh);

        preNorm_ff = new PreNorm<T>(master, feedForward, dim_model,
                prefix_str+"."+LN_ff_str, LN_gamma_str, LN_beta_str);
        residual_ff = new Residual<T>(preNorm_ff);
    }

    uint64_t parameterCount() override {
        uint64_t ret = 0;
        if (multiheadAttention) {
            ret += multiheadAttention->parameterCount();
        }
        if (feedForward) {
            ret += feedForward->parameterCount();
        }
        if (preNorm_mh) {
            ret += preNorm_mh->parameterCount();
        }
        if (preNorm_ff) {
            ret += preNorm_ff->parameterCount();
        }
        if (residual_mh) {
            ret += residual_mh->parameterCount();
        }
        if (residual_ff) {
            ret += residual_ff->parameterCount();
        }
        return ret;
    }

    ~EncoderLayer() {
        delete multiheadAttention;
        delete feedForward;
        delete preNorm_mh;
        delete preNorm_ff;
        delete residual_mh;
        delete residual_ff;
    }

    void forward(const Tensor<T> &input, Tensor<T> &output, const Tensor<bool> &mask) {
        Tensor<T> mh2ff{};
        residual_mh->forward(input, mh2ff, mask, *blank_mem);
        residual_ff->forward(mh2ff, output, *blank_mask, *blank_mem);
    }

private:
    MultiheadAttention<T> *multiheadAttention = nullptr;
    FeedForward<T> *feedForward = nullptr;
    PreNorm<T> *preNorm_mh = nullptr; 
    PreNorm<T> *preNorm_ff = nullptr;
    Residual<T> *residual_mh = nullptr;
    Residual<T> *residual_ff = nullptr;
    Tensor<bool> *blank_mask = nullptr;
    Tensor<T> *blank_mem = nullptr;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_ENCODER_LAYER_H
