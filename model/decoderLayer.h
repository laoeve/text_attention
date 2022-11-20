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

#ifndef ATTENTION_TRANSFORMER_CPP_DECODER_LAYER_H
#define ATTENTION_TRANSFORMER_CPP_DECODER_LAYER_H

#include "bits/stdc++.h"
#include "top_model.h"
#include "layer.h"
#include "residual.h"
#include "layer_norm.h"
#include "post_norm.h"
#include "multiheadattention.h"
#include "feed_forward.h"

using namespace std;

namespace text_attention {
template<typename T>
class DecoderLayer : virtual public Layer<T> {
public:
    DecoderLayer(TopModel<T>* master,
            int dim_model, int num_heads, int dim_ff, 
            const string prefix_dec, const string prefix_layer, int id, 
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
        string prefix_str = prefix_dec+"."+prefix_layer+"."+to_string(id);
        std::cout << "Init decoder - " << prefix_str << std::endl;

        /* Init attention layers */
        maskedMultiheadAttention = new MultiheadAttention<T>(master, 
                dim_model, num_heads, prefix_str, weight_str, bias_str, 
                eda_query_str, eda_key_str, eda_value_str, eda_out_str);

        multiheadAttention = new MultiheadAttention<T>(master, 
                dim_model, num_heads, prefix_str, weight_str, bias_str, 
                sa_query_str, sa_key_str, sa_value_str, sa_out_str);

        /* Init feedforward layer (MLP) */
        positionwisefeedForward = new FeedForward<T>(master, dim_model, dim_ff,
                prefix_str, weight_str, bias_str, ff_hidden_str, ff_out_str);

        /* Init layer normalizations */
        postNorm_mmh = new PostNorm<T>(master, maskedMultiheadAttention, 
                dim_model, prefix_str+"."+LN_mmh_str, LN_gamma_str, LN_beta_str);
        residual_mmh = new Residual<T>(postNorm_mmh);
        
        postNorm_mh = new PostNorm<T>(master, multiheadAttention, 
                dim_model, prefix_str+"."+LN_mh_str, LN_gamma_str, LN_beta_str);
        residual_mh = new Residual<T>(postNorm_mh);
        
        postNorm_ff = new PostNorm<T>(master, positionwisefeedForward, 
                dim_model, prefix_str+"."+LN_ff_str, LN_gamma_str, LN_beta_str);
        residual_ff = new Residual<T>(postNorm_ff);
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
        if (postNorm_mmh) {
            ret += postNorm_mmh->parameterCount();
        }
        if (postNorm_mh) {
            ret += postNorm_mh->parameterCount();
        }
        if (postNorm_ff) {
            ret += postNorm_ff->parameterCount();
        }            
        if (residual_mmh) {
            ret += residual_mmh->parameterCount();
        }
        if (residual_mh) {
            ret += residual_mh->parameterCount();
        }
        if (residual_ff) {
            ret += residual_ff->parameterCount();
        }            
        return ret;
    }

    ~DecoderLayer() {
        delete maskedMultiheadAttention;
        delete multiheadAttention;
        delete positionwisefeedForward;
        delete postNorm_mmh;
        delete postNorm_mh;
        delete postNorm_ff;
        delete residual_mmh;
        delete residual_mh;
        delete residual_ff;
    }

    void forward(Tensor<T> &output, const Tensor<T> &input, 
            const Tensor<T> &memory, const Tensor<bool> &tgt_mask, 
            const Tensor<bool> &src_mask) {
        Tensor<T> mmh2mh{ };
        Tensor<T> mh2ff{ };
        residual_mmh->forward(mmh2mh, input, tgt_mask, blank_mem);
        residual_mh->forward(mh2ff, mmh2mh, src_mask, memory);
        residual_ff->forward(output, mh2ff, blank_mask, blank_mem);
    }

private:
    MultiheadAttention<T> *maskedMultiheadAttention = nullptr;
    MultiheadAttention<T> *multiheadAttention = nullptr;
    FeedForward<T> *positionwisefeedForward = nullptr;
    PostNorm<T> *postNorm_mmh = nullptr;
    PostNorm<T> *postNorm_mh = nullptr;
    PostNorm<T> *postNorm_ff = nullptr;
    Residual<T> *residual_mmh = nullptr;
    Residual<T> *residual_mh = nullptr;
    Residual<T> *residual_ff = nullptr;
    Tensor<bool> blank_mask {};
    Tensor<T> blank_mem {};
};
}
#endif //ATTENTION_TRANSFORMER_CPP_DECODER_LAYER_H
