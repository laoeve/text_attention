//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// swin_transformer.h

#ifndef ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H
#define ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H

#include <bits/stdc++.h>

#include "top_model.h"
#include "embedding.h"
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
    AttentionTransformer(int voca_src_size, int voca_tgt_size, string model_arg)
    : voca_src_size(voca_src_size),voca_tgt_size(voca_tgt_size),model_arg(model_arg)
    {
        /* Template: string keys */
        TopModel<T>::num_layers = 6;
        TopModel<T>::dim_embed = 512;
        TopModel<T>::num_heads = 8;
        TopModel<T>::dim_ff = 2048;
        int num_layers = TopModel<T>::num_layers; 
        int dim_embed = TopModel<T>::dim_embed; 
        int num_heads = TopModel<T>::num_heads;
        int dim_ff = TopModel<T>::dim_ff; 

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

        const string prefix_em_src = "src_embed";
        const string prefix_em_tgt = "tgt_embed";
        const string em_str = "0.lut.weight";
        const string pe_str = "1.pe";
        const string gen_str = "generator.proj";

        /* Init embedding layers */
        Tensor<T>* lut_em_src = new Tensor<T>(
                param_map[prefix_em_src+"."+em_str].pvals,
                param_map[prefix_em_src+"."+em_str].pshape);
        Tensor<T>* lut_pe_src = new Tensor<T>(
                param_map[prefix_em_src+"."+pe_str].pvals,
                param_map[prefix_em_src+"."+pe_str].pshape);

        embed_src = new Embedding<T>(prefix_em_src, dim_embed,
                *lut_em_src, *lut_pe_src);

        Tensor<T>* lut_em_tgt = new Tensor<T>(
                param_map[prefix_em_tgt+"."+em_str].pvals,
                param_map[prefix_em_tgt+"."+em_str].pshape);
        Tensor<T>* lut_pe_tgt = new Tensor<T>(
                param_map[prefix_em_tgt+"."+pe_str].pvals,
                param_map[prefix_em_tgt+"."+pe_str].pshape);

        embed_tgt = new Embedding<T>(prefix_em_tgt, dim_embed,
                *lut_em_tgt, *lut_pe_tgt);

        embed_src->print_params( );
        embed_tgt->print_params( );

        /* Init encoder/decoder layers */
        encoder = new Encoder<T>(this, num_layers, dim_embed, num_heads, 
                dim_ff, prefix_enc, prefix_layer, weight_str, bias_str, 
                LN_gamma_str, LN_beta_str, sa_query_str, 
                sa_key_str, sa_value_str, sa_out_str, 
                ff_hidden_str, ff_out_str, LN_mh_str, LN_ff_str);
        decoder = new Decoder<T>(this, num_layers, dim_embed, num_heads, 
                dim_ff, prefix_dec, prefix_layer, weight_str, bias_str, 
                LN_gamma_str, LN_beta_str, sa_query_str, 
                sa_key_str, sa_value_str, sa_out_str,
                eda_query_str, eda_key_str, eda_value_str, eda_out_str,
                ff_hidden_str, ff_out_str, 
                LN_dec_mmh_str, LN_dec_mh_str, LN_dec_ff_str);

        vector<T>* gamma = new vector<T>(param_map[prefix_enc+"."+LN_gamma_str].pvals);
        vector<T>* beta = new vector<T>(param_map[prefix_enc+"."+LN_beta_str].pvals);
        ln_encoder = new LayerNorm<T>(prefix_enc, dim_embed, *gamma, *beta);

        gamma = new vector<T>(param_map[prefix_dec+"."+LN_gamma_str].pvals);
        beta = new vector<T>(param_map[prefix_dec+"."+LN_beta_str].pvals);
        ln_decoder = new LayerNorm<T>(prefix_dec, dim_embed, *gamma, *beta);

        /* Init generator layer */
        Tensor<T>* gen_w = new Tensor<T>(
                param_map[gen_str+"."+weight_str].pvals,
                param_map[gen_str+"."+weight_str].pshape);
        Tensor<T>* gen_b = new Tensor<T>(
                param_map[gen_str+"."+weight_str].pvals,
                param_map[gen_str+"."+weight_str].pshape);

        generator = new Linear<T>(gen_str, dim_embed, 
                voca_tgt_size, *gen_w, *gen_b);

        generator->print_params( );
    }

    void forward(const Tensor<T> &input, Tensor<T> &output) override 
    {
        Tensor<T> enc_out_inter{};  // intermediate output tensor from encoder
        Tensor<T> enc_out_fin{};    // final output tensor from encoder LN
        Tensor<T> input_embed{};

        /* Setup encoder mask */
        Tensor<bool> src_mask{};
        TopModel<T>::set_enc_mask(input, src_mask);

        input.print_all( );
        src_mask.print_all( );

        /* Encoder forward */
        embed_src->forward(input_embed, input);
        encoder->forward(enc_out_inter, input_embed, src_mask);
        ln_encoder->forward(enc_out_fin, enc_out_inter);

        /* Decoder part operation word-by-word */
        Tensor<T> tgt_input(vector<int>{input.shape[0], 2});
        tgt_input[0] = 1; // <start of sentence>
        for (int i=0; i<SENTENCE_LEN-2; i++)
        {
            /* Setup target mask */
            Tensor<bool> enc_mask{};
            Tensor<bool> tgt_mask{};
            TopModel<T>::set_pad_mask(enc_mask, tgt_input, input);
            TopModel<T>::set_dec_mask(tgt_mask, tgt_input);

            /* Decoder forward */
            Tensor<T> tgt_embed{ };
            Tensor<T> dec_out_inter{ }; // intermediate output tensor from decoder
            Tensor<T> dec_out_fin{ };   // final output tensor from decoder LN
            embed_tgt->forward(tgt_embed, tgt_input);
            decoder->forward(dec_out_inter, tgt_embed, enc_out_fin, tgt_mask, enc_mask);
            ln_decoder->forward(dec_out_fin, dec_out_inter);

            /* Generator and softmax */
            Tensor<T> gen_out{ };
            Tensor<T> sm_out{ };
            generator->forward(gen_out, dec_out_fin);
            softMax.forward(sm_out, gen_out);

            /* Find max value of probability */
            Tensor<T> max_indices{ };
            max_tensor.forward(max_indices, sm_out);

            /* Set indices across the batch */
            set_new_tgt_input(tgt_input, max_indices, i);
            std::cout << " Print : Whole Sentence > ";
            max_indices.print_all();
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
    int SENTENCE_LEN;
    string model_arg;
    Embedding<T>* embed_src = nullptr;
    Embedding<T>* embed_tgt = nullptr;
    Encoder<T> *encoder = nullptr;
    Decoder<T> *decoder = nullptr;
    LayerNorm<T>* ln_encoder = nullptr;
    LayerNorm<T>* ln_decoder = nullptr;
    Linear<T> *generator = nullptr;
    SoftMax<T> softMax;
};
}

#endif //ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H
