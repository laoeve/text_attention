/*
 * Copyright (c) 2022 Computer Architecture and Paralllel Processing Lab, 
 * Seoul National University, Republic of Korea. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     1. Redistribution of source code must retain the above copyright 
 *        notice, this list of conditions and the follwoing disclaimer.
 *     2. Redistributions in binary form must reproduce the above copyright 
 *        notice, this list conditions and the following disclaimer in the 
 *        documentation and/or other materials provided with the distirubtion.
 *     3. Neither the name of the copyright holders nor the name of its 
 *        contributors may be used to endorse or promote products derived from 
 *        this software without specific prior written permission. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 
 * Hyokeun Lee (hklee@capp.snu.ac.kr)
 * Hyunjun Park (laoeve@capp.snu.ac.kr)
 *
 */

#ifndef ATTENTION_TRANSFORMER_CPP_GPT2_H
#define ATTENTION_TRANSFORMER_CPP_GPT2_H

#include <bits/stdc++.h>

#include "top_model.h"
#include "embedding.h"
#include "layer.h"
#include "tensor.h"
#include "decoder_gpt2.h"
#include "layer_norm.h"
#include "softmax.h"
#include "linear.h"
#include "max_tensor.h"

using namespace std;

namespace text_attention {
template<typename T>
class GPT2 : virtual public TopModel<T> 
{
public:
    GPT2(int voca_src_size, int voca_tgt_size, string model_arg)
    : voca_src_size(voca_src_size),voca_tgt_size(voca_tgt_size),model_arg(model_arg)
    {
        /* Template */
        SENTENCE_LEN = 32;
        TopModel<T>::num_layers = 12;
        TopModel<T>::dim_embed = 768;
        TopModel<T>::num_heads = 12;
        TopModel<T>::dim_ff = 3072;
        int num_layers = TopModel<T>::num_layers; 
        int dim_embed = TopModel<T>::dim_embed; 
        int num_heads = TopModel<T>::num_heads;
        int dim_ff = TopModel<T>::dim_ff; 

        const string prefix_enc = "";
        const string prefix_dec = "h";
        const string prefix_layer = "";
        const string weight_str = "weight";
        const string bias_str = "bias";

        const string sa_query_str = "attn.c_attn"; // self-attention
        const string sa_key_str = "attn.c_attn";
        const string sa_value_str = "attn.c_attn";
        const string sa_out_str = "attn.c_proj";
        const string eda_query_str = "";
        const string eda_key_str = "";
        const string eda_value_str = "";
        const string eda_out_str = "";

        const string ff_hidden_str = "mlp.c_fc";
        const string ff_out_str = "mlp.c_proj";

        const string LN_mh_str = "";
        const string LN_ff_str = "";
        const string LN_dec_mmh_str = "";
        const string LN_dec_mh_str = "ln_1";
        const string LN_dec_ff_str = "ln_2";
        const string LN_out = "";
        const string LN_gamma_str = "weight";
        const string LN_beta_str = "bias";

        const string prefix_em_token = "wte";
        const string prefix_em_pos = "wpe";
        const string em_str = "weight";
        const string pe_str = "weight";
        const string ln_f_str = "ln_f";

        /* Init embedding layers */
        lut_em_token = new Tensor<T>(
                param_map[prefix_em_token+"."+em_str].pvals,
                param_map[prefix_em_token+"."+em_str].pshape);
        Tensor<T>* lut_em_pos = new Tensor<T>(
                param_map[prefix_em_pos+"."+pe_str].pvals,
                param_map[prefix_em_pos+"."+pe_str].pshape);

        embed_words = new Embedding<T>(prefix_em_token, dim_embed,
                *lut_em_token, *lut_em_pos);

        /* Init encoder/decoder layers */
        decoder_gpt2 = new Decoder_GPT2<T>(this, num_layers, dim_embed, num_heads, 
                dim_ff, prefix_dec, prefix_layer, weight_str, bias_str, 
                LN_gamma_str, LN_beta_str, sa_query_str, 
                sa_key_str, sa_value_str, sa_out_str,
                ff_hidden_str, ff_out_str, LN_dec_mh_str, LN_dec_ff_str);

        /* Init ln_final layer */
        Tensor<T>* gamma = new Tensor<T>(
                param_map[ln_f_str+"."+weight_str].pvals,
                param_map[ln_f_str+"."+weight_str].pshape);
        Tensor<T>* beta = new Tensor<T>(
                param_map[ln_f_str+"."+bias_str].pvals,
                param_map[ln_f_str+"."+bias_str].pshape);

        ln_final = new LayerNorm<T>(ln_f_str, dim_embed, *gamma, *beta);
    }

    void forward(Tensor<T> &output, const Tensor<T> &input) override 
    {
        Tensor<T> dec_out_inter{};  // intermediate output tensor from encoder
        Tensor<T> dec_out_fin{};    // final output tensor from encoder LN
        Tensor<T> input_embed{};

        Tensor<T> input_iter(input, input.shape);

        for (int i=0; i<SENTENCE_LEN-2; i++)
        {
            /* Setup encoder mask */
            Tensor<bool> tgt_mask{};
            TopModel<T>::set_dec_mask(tgt_mask, input_iter);

            embed_words->forward(input_embed, input_iter);

#ifdef DEBUG
            std::cout << "Generating the word at " << i+1 << std::endl;
#endif

            decoder_gpt2->forward(dec_out_inter, input_embed, tgt_mask);

            /* ln_final and softmax */
            Tensor<T> ln_out{ };
            ln_final->forward(ln_out, dec_out_inter);
        
            //Layer<T>* layer_tensor { };
            Tensor<T> logits{ };
            //gen_out.reshape(std::vector(gen_out.shape[1], gen_out.shape[2]));
            layer_tensor->matmul(logits, ln_out, *lut_em_token, 1.0);

            /* Find max value of probability */
            Tensor<T> max_indices{ };
            max_tensor.forward(max_indices, logits);
        }
    }

    uint64_t parameterCount() 
    {
        return decoder_gpt2->parameterCount() +
               ln_final->parameterCount();
    }

private:
    void set_new_tgt_input(Tensor<T>& tgt_input, 
            const Tensor<T>& max_indices, const int widx)
    {
        int new_len = tgt_input.shape[1]+1;
        int num_input = tgt_input.shape[0];
        Tensor<T> tmp_tgt_input(vector<int>{num_input, new_len});
        for (int n=0; n<num_input; n++)
        {
            for (int j=0; j<tgt_input.shape[1]; j++)
            {
                tmp_tgt_input[n*new_len+j] = 
                    tgt_input[n*tgt_input.shape[1]+j];
            }

            tmp_tgt_input[n*new_len+widx+1] = 
                max_indices[n*max_indices.shape[1]+widx+1];
        }
        tgt_input = tmp_tgt_input;
    }

    int voca_src_size;
    int voca_tgt_size;
    int SENTENCE_LEN;
    string model_arg;
    Tensor<T>* lut_em_token = nullptr;
    Embedding<T>* embed_words = nullptr;
    Decoder_GPT2<T> *decoder_gpt2 = nullptr;
    LayerNorm<T> *ln_final = nullptr;
    MaxTensor<T> max_tensor;
    Layer<T> *layer_tensor;
};
}

#endif //ATTENTION_TRANSFORMER_CPP_GPT2_H
