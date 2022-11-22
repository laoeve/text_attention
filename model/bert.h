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

#ifndef BERT_BASE_H
#define BERT_BASE_H

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
#include "max_tensor.h"

using namespace std;

namespace text_attention {
template<typename T>
class BERT : virtual public TopModel<T> 
{
public:
    BERT(int voca_src_size, int voca_tgt_size, string model_arg)
    : voca_src_size(voca_src_size),voca_tgt_size(voca_tgt_size),model_arg(model_arg)
    {
        /* Template */
        SENTENCE_LEN = 32;     // equivalent iteration value for comparison
        if (model_arg == "bert-base")
        {
                TopModel<T>::num_layers = 12;
                TopModel<T>::dim_embed = 768;
                TopModel<T>::num_heads = 12;
                TopModel<T>::dim_ff = 3072;
        }
        else if (model_arg == "bert-large")
        {
                TopModel<T>::num_layers = 24;
                TopModel<T>::dim_embed = 1024;
                TopModel<T>::num_heads = 16;
                TopModel<T>::dim_ff = 4096;
        }
        else
        {
                cout << "error : BERT Model not matched" << endl;
                assert(0);
        }
        int num_layers = TopModel<T>::num_layers; 
        int dim_embed = TopModel<T>::dim_embed; 
        int num_heads = TopModel<T>::num_heads;
        int dim_ff = TopModel<T>::dim_ff; 

        const string prefix_enc = "encoder";
        const string prefix_layer = "layer";
        const string prefix_embed = "embed";
        const string weight_str = "weight";
        const string bias_str = "bias";

        const string sa_query_str = "attention.self.query"; // self-attention
        const string sa_key_str = "attention.self.key";
        const string sa_value_str = "attention.self.value";
        const string sa_out_str = "attention.output.dense";

        const string ff_hidden_str = "intermediate.dense";
        const string ff_out_str = "output.dense";

        const string LN_mh_str = "attention.output.LayerNorm";
        const string LN_ff_str = "output.LayerNorm";
        const string LN_out = "LayerNorm";
        const string LN_gamma_str = "weight";
        const string LN_beta_str = "bias";

        const string prefix_em_src = "embeddings";
        const string prefix_em_tgt = "";
        const string em_str = "word_embeddings.weight";
        const string pe_str = "position_embeddings.weight";
        const string embed_norm_str = "embeddings.LayerNorm";
        const string pooler_str = "pooler.dense";

        param_map[prefix_em_src+"."+em_str].pvals.size();
        param_map[prefix_em_src+"."+em_str].pshape.size();

        /* Init embedding layers */
        Tensor<T>* lut_em_src = new Tensor<T>(
                param_map[prefix_em_src+"."+em_str].pvals,
                param_map[prefix_em_src+"."+em_str].pshape);
        Tensor<T>* lut_pe_src = new Tensor<T>(
                param_map[prefix_em_src+"."+pe_str].pvals,
                param_map[prefix_em_src+"."+pe_str].pshape);
        
        embed_src = new Embedding<T>(prefix_em_src, dim_embed,
                *lut_em_src, *lut_pe_src);

        vector<T>* gamma = new vector<T>(
                param_map[prefix_em_src+"."+LN_out+"."+LN_gamma_str].pvals);
        vector<T>* beta = new vector<T>(
                param_map[prefix_em_src+"."+LN_out+"."+LN_beta_str].pvals);
        norm_embed_src = new LayerNorm<T>(prefix_em_src+"."+LN_out, dim_embed, *gamma, *beta);

        embed_src->print_params( );

        /* Init encoder/decoder layers */
        encoder = new Encoder<T>(this, num_layers, dim_embed, num_heads, 
                dim_ff, prefix_enc, prefix_layer, weight_str, bias_str, 
                LN_gamma_str, LN_beta_str, sa_query_str, 
                sa_key_str, sa_value_str, sa_out_str, 
                ff_hidden_str, ff_out_str, LN_mh_str, LN_ff_str);

        /* Init generator layer */
        Tensor<T>* pool_w = new Tensor<T>(
                param_map[pooler_str+"."+weight_str].pvals,
                param_map[pooler_str+"."+weight_str].pshape);
        Tensor<T>* pool_b = new Tensor<T>(
                param_map[pooler_str+"."+bias_str].pvals,
                param_map[pooler_str+"."+bias_str].pshape);

        pooler = new Linear<T>(pooler_str, dim_embed, 
                dim_embed, *pool_w, *pool_b);

        pooler->print_params( );
    }

    void forward(Tensor<T> &output, const Tensor<T> &input) override 
    {
        Tensor<T> input_iter(input, input.shape);

        Tensor<T> enc_out_inter{};  // intermediate output tensor from encoder
        Tensor<T> enc_out_fin{};    // final output tensor from encoder LN
        Tensor<T> input_embed{};
        Tensor<T> input_norm{};

        /* Setup encoder mask */
        Tensor<bool> src_mask{};
        TopModel<T>::set_pad_mask(src_mask, input_iter, input_iter);


        for (int i=0; i<SENTENCE_LEN-2; i++)
        {
#ifdef DEBUG
            std::cout << "Generating the word at " << i+1 << std::endl;
#endif

            /* Encoder forward */
            embed_src->forward(input_embed, input_iter);
            norm_embed_src->forward(input_norm, input_embed);
            
            encoder->forward(enc_out_fin, input_norm, src_mask);  //To-Do : change tgt_mask
            
            Tensor<T> pooler_out{ };
            pooler->forward(pooler_out, enc_out_fin);
            
        }
    }

    uint64_t parameterCount() 
    {
        return norm_embed_src->parameterCount() + encoder->parameterCount() +
               pooler->parameterCount();
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
    Embedding<T>* embed_src = nullptr;
    Encoder<T> *encoder = nullptr;
    LayerNorm<T> *norm_embed_src = nullptr;
    Linear<T> *pooler = nullptr;
    MaxTensor<T> max_tensor;
};
}

#endif //ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H
