//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// based from swin_block.h

#ifndef ATTENTION_TRANSFORMER_CPP_DECODER_LAYER_H
#define ATTENTION_TRANSFORMER_CPP_DECODER_LAYER_H

#include "layer.h"
#include "residual.h"
#include "layer_norm.h"
#include "multiheadattention.h"
#include "feed_forward.h"

namespace text_attention {
template<typename T>
class DecoderLayer : virtual public Layer<T> {
public:
    DecoderLayer(int num_layer, int dim_model, int dim_ff, 
            int heads, int max_len, std::string str_key_layer) {
        maskedMultiheadAttention = new MultiheadAttention<T>(
                dim_model, heads, max_len, str_key_layer + "self_attn.linears.");

        multiheadAttention = new MultiheadAttention<T>(dim_model, 
                heads, max_len, str_key_layer + "src_attn.linears.");

        positionwisefeedForward = new FeedForward<T>(
                dim_model, dim_ff, str_key_layer + "feed_forward.");

        preNorm1 = new PreNorm<T>(maskedMultiheadAttention, 
                dim_model, str_key_layer + "sublayer.0.");
        preNorm2 = new PreNorm<T>(multiheadAttention, 
                dim_model, str_key_layer + "sublayer.1.");
        preNorm3 = new PreNorm<T>(positionwisefeedForward, 
                dim_model, str_key_layer + "sublayer.2.");

        residual1 = new Residual<T>(preNorm1);
        residual2 = new Residual<T>(preNorm2);
        residual3 = new Residual<T>(preNorm3);
    }

    uint64_t parameterCount() override {
        uint64_t ret = 0;
        if (maskedMultiheadAttention) {
            ret += maskedMultiheadAttention->parameterCount();
        }
        if (multiheadAttention) {
            ret += multiheadAttention->parameterCount();
        }            
        if (positionwisefeedForward) {
            ret += positionwisefeedForward->parameterCount();
        }
        if (preNorm1) {
            ret += preNorm1->parameterCount();
        }
        if (preNorm2) {
            ret += preNorm2->parameterCount();
        }
        if (preNorm3) {
            ret += preNorm3->parameterCount();
        }            
        if (residual1) {
            ret += residual1->parameterCount();
        }
        if (residual2) {
            ret += residual2->parameterCount();
        }
        if (residual3) {
            ret += residual3->parameterCount();
        }            
        return ret;
    }

    ~DecoderLayer() {
        if (maskedMultiheadAttention != nullptr) {
            delete maskedMultiheadAttention;
            maskedMultiheadAttention = nullptr;
        }
        if (multiheadAttention != nullptr) {
            delete multiheadAttention;
            multiheadAttention = nullptr;
        }            
        if (positionwisefeedForward != nullptr) {
            delete positionwisefeedForward;
            positionwisefeedForward = nullptr;
        }
        if (preNorm1 != nullptr) {
            delete preNorm1;
            preNorm1 = nullptr;
        }
        if (preNorm2 != nullptr) {
            delete preNorm2;
            preNorm2 = nullptr;
        }
        if (preNorm3 != nullptr) {
            delete preNorm3;
            preNorm3 = nullptr;
        }            
        if (residual1 != nullptr) {
            delete residual1;
            residual1 = nullptr;
        }
        if (residual2 != nullptr) {
            delete residual2;
            residual2 = nullptr;
        }
        if (residual3 != nullptr) {
            delete residual3;
            residual3 = nullptr;
        }            
    }

    void forward(const Tensor<T> &input, Tensor<T> &memory, 
            Tensor<T> &output, Tensor<T> &tgt_mask, Tensor<T> &src_mask) {
        Tensor<T> tmp1{};
        Tensor<T> tmp2{};
        residual1->forward(input, tmp1, tgt_mask, *blank);
        residual2->forward(tmp1, tmp2, src_mask, memory);
        residual3->forward(tmp2, output, *blank, *blank);
    }

private:
    Residual<T> *residual1 = nullptr, *residual2 = nullptr, *residual3 = nullptr;
    PreNorm<T> *preNorm1 = nullptr, *preNorm2 = nullptr, *preNorm3 = nullptr;
    MultiheadAttention<T> *maskedMultiheadAttention = nullptr;
    MultiheadAttention<T> *multiheadAttention = nullptr;
    FeedForward<T> *positionwisefeedForward = nullptr;
    Tensor<T> *blank = nullptr;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_DECODER_LAYER_H
