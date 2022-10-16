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
//#include "patch_merging.h"
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
    
    // changed from w-msa
    void forward(const Tensor<T> &input, Tensor<T> &output, 
            Tensor<T> &memory, Tensor<T> &tgt_mask, Tensor<T> &src_mask) { 
        Tensor<T> tmp{};
        int layer_num = 0;
        tmp = input;
        for (auto blockPtr: layers) {
            std::cout << "Fwd Decoder." << layer_num++ << std::endl;
            Tensor<T> tmp_loop{};
            blockPtr->forward(tmp, memory, tmp_loop, tgt_mask, src_mask);
            tmp = tmp_loop;
        }
        output.clear();
        output.shape.clear();
        output.insert(output.end(), tmp.begin(), tmp.end());
        output.shape.insert(output.shape.end(), tmp.shape.begin(), tmp.shape.end());

        /*
        Tensor<T> tmp{};
        Tensor<T> tmp1{};
        Tensor<T> tmp2{};
        Tensor<T> tmp3{};
        Tensor<T> tmp4{};
        Tensor<T> tmp5{};
        Tensor<T> tmp6{};
        Tensor<T> tmp7{};
        enc1.forward(input, tmp1);
        enc2.forward(tmp1, tmp2);
        enc3.forward(tmp2, tmp3);
        enc4.forward(tmp3, tmp4);
        enc5.forward(tmp4, tmp5);
        enc6.forward(tmp5, tmp6);   
        layerNorm.forward(tmp6, tmp7);
        */

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
