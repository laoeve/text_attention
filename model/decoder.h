//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// based from stage_module.h
// 

#ifndef ATTENTION_TRANSFORMER_CPP_DECODER_H
#define ATTENTION_TRANSFORMER_CPP_DECODER_H

#include <vector>

#include "top_model.h"
#include "tensor.h"
#include "layer.h"
#include "decoderLayer.h"

using namespace std;

namespace text_attention {
template<typename T>
class Decoder : public Layer<T> {
public:
    Decoder(TopModel<T>* master,
            int num_layers, int dim_model, int num_heads, int dim_ff,
            const string prefix_dec, const string prefix_layer,
            const string weight_str, const string bias_str, 
            const string LN_gamma_str, const string LN_beta_str,
            const string sa_query_str, const string sa_key_str,
            const string sa_value_str, const string sa_out_str,
            const string eda_query_str, const string eda_key_str,
            const string eda_value_str, const string eda_out_str,
            const string ff_hidden_str, const string ff_out_str,
            const string LN_mmh_str, const string LN_mh_str, const string LN_ff_str) 
    : Layer<T>(master)
    {
        for (int id=0; id<num_layers; id++) 
        {
            DecoderLayer<T>* dec = new DecoderLayer<T>(master,
                    dim_model, num_heads, dim_ff, prefix_dec, prefix_layer, id,
                    weight_str, bias_str, LN_gamma_str, LN_beta_str,
                    sa_query_str, sa_key_str, sa_value_str, sa_out_str,
                    eda_query_str, eda_key_str, eda_value_str, eda_out_str,
                    ff_hidden_str, ff_out_str, LN_mmh_str, LN_mh_str, LN_ff_str);
            layers.push_back(dec);
        }
    }

    ~Decoder() {
        for (int i = 0; i < layers.size(); ++i) {
            delete layers[i];
        }
    }
    
    void forward(const Tensor<T> &input, Tensor<T> &output, 
            const Tensor<T> &memory, const Tensor<bool> &tgt_mask, 
            const Tensor<bool> &src_mask) { 
        Tensor<T> tmp_in(input);
        int layer_num = 0;
        for (auto blockPtr: layers) 
        {
#ifdef DEBUG
            std::cout << "Forward pass of decoder[" << layer_num++ << "]" << std::endl;
#endif
            layer_num = layer_num;

            blockPtr->forward(tmp_in, output, memory, tgt_mask, src_mask);
            tmp_in = output;
        }
    }

    uint64_t parameterCount() override {
        uint64_t ret=0;
        for (int i = 0; i < layers.size(); ++i) {
            ret += layers[i]->parameterCount();
        }
        return ret;
    }

private:
    std::vector<DecoderLayer<T> *> layers;
};
}


#endif //ATTENTION_TRANSFORMER_CPP_DECODER_H