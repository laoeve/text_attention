//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// based from swin_block.h

#ifndef ATTENTION_TRANSFORMER_CPP_ENCODER_LAYER_H
#define ATTENTION_TRANSFORMER_CPP_ENCODER_LAYER_H

#include "layer.h"
#include "residual.h"
#include "pre_norm.h"
#include "multiheadattention.h"
#include "feed_forward.h"

namespace text_attention {
    template<typename T>
    class EncoderLayer : virtual public Layer<T> {
    public:
        EncoderLayer(int num_layer, int dim_model, int dim_ff, int heads, int max_len, std::string str_key_layer) {
            multiheadAttention = new MultiheadAttention<T>(dim_model, heads, max_len, str_key_layer + "self_attn.linears.");
            feedForward = new FeedForward<T>(dim_model, dim_ff, str_key_layer + "feed_forward.");
            preNorm1 = new PreNorm<T>(multiheadAttention, dim_model, str_key_layer + "sublayer.0.");
            preNorm2 = new PreNorm<T>(feedForward, dim_model, str_key_layer + "sublayer.1.");
            residual1 = new Residual<T>(preNorm1);
            residual2 = new Residual<T>(preNorm2);
        }   // hjpark) dropout not included

        long long parameterCount() {
            long long ret = 0;
            if (multiheadAttention) {
                ret += multiheadAttention->parameterCount();
            }
            if (feedForward) {
                ret += feedForward->parameterCount();
            }
            if (preNorm1) {
                ret += preNorm1->parameterCount();
            }
            if (preNorm2) {
                ret += preNorm2->parameterCount();
            }
            if (residual1) {
                ret += residual1->parameterCount();
            }
            if (residual2) {
                ret += residual2->parameterCount();
            }
            return ret;
        }

        ~EncoderLayer() {
            if (multiheadAttention != nullptr) {
                delete multiheadAttention;
                multiheadAttention = nullptr;
            }
            if (feedForward != nullptr) {
                delete feedForward;
                feedForward = nullptr;
            }
            if (preNorm1 != nullptr) {
                delete preNorm1;
                preNorm1 = nullptr;
            }
            if (preNorm2 != nullptr) {
                delete preNorm2;
                preNorm2 = nullptr;
            }
            if (residual1 != nullptr) {
                delete residual1;
                residual1 = nullptr;
            }
            if (residual2 != nullptr) {
                delete residual2;
                residual2 = nullptr;
            }
        }

        void forward(const Tensor<T> &input, Tensor<T> &output, Tensor<T> &mask) {
            Tensor<T> tmp{};
            
            residual1->forward(input, tmp, mask, *blank);
            residual2->forward(tmp, output, *blank, *blank);

        }

    private:
        Residual<T> *residual1 = nullptr, *residual2 = nullptr;
        PreNorm<T> *preNorm1 = nullptr, *preNorm2 = nullptr;
        MultiheadAttention<T> *multiheadAttention = nullptr;
        FeedForward<T> *feedForward = nullptr;
        Tensor<T> *blank = nullptr;
    };
}
#endif //ATTENTION_TRANSFORMER_CPP_ENCODER_LAYER_H
