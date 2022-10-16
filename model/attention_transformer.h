//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// swin_transformer.h

#ifndef ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H
#define ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H

#include <bits/stdc++.h>

#include "top_model.h"
#include "layer.h"
#include "tensor.h"
#include "encoder.h"
#include "decoder.h"
#include "layer_norm.h"
#include "softmax.h"
#include "linear.h"

using namespace std;

namespace text_attention {
template<typename T>
class AttentionTransformer : virtual public TopModel<T> 
{
public:
    AttentionTransformer(int num_layers, int dim_model, int num_heads, 
            int dim_ff, int voca_src_size, int voca_tgt_size)
    : voca_src_size(voca_src_size),voca_tgt_size(voca_tgt_size)
    {
        /* Template: string keys */
        const string prefix_enc = "encoder";
        const string prefix_dec = "decoder";
        const string prefix_layer = "layers";
        const string weight_str = "weight";
        const string bias_str = "bias";

        const string sa_query_str = "self_attn.linears.0"; // self-attention
        const string sa_key_str = "self_attn.linears.1";
        const string sa_value_str = "self_attn.linears.2";
        const string sa_out_str = "self_attn.linears.3";
        const string eda_query_str = "src_attn.linears.0"; // enc-dec attention
        const string eda_key_str = "src_attn.linears.1";
        const string eda_value_str = "src_attn.linears.2";
        const string eda_out_str = "src_attn.linears.3";

        const string ff_hidden_str = "feed_forward.w_1";
        const string ff_out_str = "feed_forward.w_2";

        const string LN_mh_str = "sublayer.0.norm";
        const string LN_ff_str = "sublayer.1.norm";
        const string LN_dec_mmh_str = "sublayer.0.norm";
        const string LN_dec_mh_str = "sublayer.1.norm";
        const string LN_dec_ff_str = "sublayer.2.norm";
        const string LN_gamma_str = "a_2";
        const string LN_beta_str = "b_2";

        const string gen_str = "generator.proj";

        /* Init model dimension parameters */
        this->num_layers = num_layers;
        this->dim_embed = dim_model;
        this->num_heads = num_heads;
        this->dim_ff = dim_ff;

        encoder = new Encoder<T>(this, num_layers, dim_model, num_heads, 
                dim_ff, prefix_enc, prefix_layer, weight_str, bias_str, 
                LN_gamma_str, LN_beta_str, sa_query_str, 
                sa_key_str, sa_value_str, sa_out_str, 
                ff_hidden_str, ff_out_str, LN_mh_str, LN_ff_str);
        decoder = new Decoder<T>(this, num_layers, dim_model, num_heads, 
                dim_ff, prefix_dec, prefix_layer, weight_str, bias_str, 
                LN_gamma_str, LN_beta_str, sa_query_str, 
                sa_key_str, sa_value_str, sa_out_str,
                eda_query_str, eda_key_str, eda_value_str, eda_out_str,
                ff_hidden_str, ff_out_str, 
                LN_dec_mmh_str, LN_dec_mh_str, LN_dec_ff_str);

        /* Get weight and bias parameters */
        if (gen_str!="")
        {
            Tensor<T>* gen_w = new Tensor<T>(
                    param_map[gen_str+"."+weight_str].pvals,
                    param_map[gen_str+"."+weight_str].pshape);
            Tensor<T>* gen_b = new Tensor<T>(
                    param_map[gen_str+"."+weight_str].pvals,
                    param_map[gen_str+"."+weight_str].pshape);

            generator = new Linear<T>(gen_str, dim_model, 
                    voca_tgt_size, *gen_w, *gen_b);

            generator->print_params( );
        }
    }

    void forward(const Tensor<T> &input, Tensor<T> &output) override {
        Tensor<T> memory{};
        Tensor<T> src_mask{};
        Tensor<T> input_embed{};
        std::vector<float> lut;

        // making encoder mask : tensor( x for x in input != pad)
        for(int i = 0; i < input.shape[1] ; i++){
            if(input[(1*i)] == 2){
                src_mask.push_back(false);
            } else {
                src_mask.push_back(true);
            }
        }
        src_mask.shape = {1, 1, input.shape[1]};

        embed_idx(input, input_embed, TopModel<T>::dim_embed, 
                "src_embed.0.lut.weight", "src_embed.1.pe");

        encoder->forward(input_embed, memory, src_mask);
        
        // encoder input : tensor([[[0]]]).embed import
        Tensor<T> tgt_input{};
        Tensor<T> tgt_embed{};
        Tensor<T> tgt_mask{};
        Tensor<T> tmp3{};
        Tensor<T> decoder_out{};

        tgt_input.push_back(0);
        tgt_input.shape={1,1};

        // decoder mask : [1,0][1,1] ...
        // encoder mask : same with encoder mask used in encoder
        // Decoder loop for expect sentence
        for(int i=0; i < max_len; i++){
            //making tgt_mask.shape={j,j};
            tgt_mask.clear();
            for(int j = 1; j < tgt_input.size() ; j++){
                tgt_mask.push_back(true);
                if(j != 1){
                    for(int m_size = 0; m_size < j; m_size++){      // [1,1,..,1,0, ..,0,0]
                        std::vector<int> m_front(m_size,1); // [1,1 ...]
                        std::vector<int> m_back(j-m_size-1,0);  // [0,0, ...]
                        m_front.insert(m_front.end(), m_back.begin(), m_back.end());    // [1, .. ,1,0, .. ,0]
                        tgt_mask.insert(tgt_mask.end(), m_front.begin(), m_front.end());
                    }
                }
            }
            tgt_mask.shape.clear();
            tgt_mask.shape = {tgt_input.size(), tgt_input.size()};

            embed_idx(tgt_input, tgt_embed, TopModel<T>::dim_embed, "tgt_embed.0.lut.weight", "tgt_embed.1.pe");

            decoder->forward(tgt_embed, decoder_out, memory, tgt_mask, src_mask);  
            generator->forward(decoder_out, tmp3);
            softMax.forward(tmp3, output);

            //find max value of probability
            int max_index = max_element(output.begin(), output.end()) - output.begin();
            std::cout << "next word : " << max_index << std::endl;
            
            tgt_input.push_back(max_index);
            tgt_input.shape = {1,tgt_input.shape[1] + 1};
        }
    }

    uint64_t parameterCount() 
    {
        return encoder->parameterCount() + decoder->parameterCount() +
               generator->parameterCount() + softMax.parameterCount();
    }

private:
    int voca_src_size;
    int voca_tgt_size;
    Encoder<T> *encoder = nullptr;
    Decoder<T> *decoder = nullptr;
    Linear<T> *generator = nullptr;
    SoftMax<T> softMax;
};
}

#endif //ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H
