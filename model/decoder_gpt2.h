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

#ifndef ATTENTION_TRANSFORMER_CPP_DECODER_GPT2_H
#define ATTENTION_TRANSFORMER_CPP_DECODER_GPT2_H

#include "bits/stdc++.h"
#include "top_model.h"
#include "tensor.h"
#include "layer.h"
#include "decoderLayer_gpt2.h"

using namespace std;

namespace text_attention {
template<typename T>
class Decoder_GPT2 : public Layer<T> {
public:
    Decoder_GPT2(TopModel<T>* master,
            int num_layers, int dim_model, int num_heads, int dim_ff,
            const string prefix_dec, const string prefix_layer,
            const string weight_str, const string bias_str, 
            const string LN_gamma_str, const string LN_beta_str,
            const string sa_query_str, const string sa_key_str,
            const string sa_value_str, const string sa_out_str,
            const string ff_hidden_str, const string ff_out_str,
            const string LN_mh_str, const string LN_ff_str) 
    : Layer<T>(master)
    {
        for (int id=0; id<num_layers; id++) 
        {
            DecoderLayer_GPT2<T>* dec_gpt2 = new DecoderLayer_GPT2<T>(master,
                    dim_model, num_heads, dim_ff, prefix_dec, prefix_layer, id,
                    weight_str, bias_str, LN_gamma_str, LN_beta_str,
                    sa_query_str, sa_key_str, sa_value_str, sa_out_str,
                    ff_hidden_str, ff_out_str, LN_mh_str, LN_ff_str);
            layers_gpt2.push_back(dec_gpt2);
        }
    }

    ~Decoder_GPT2() {
        for (int i = 0; i < layers_gpt2.size(); ++i) {
            delete layers_gpt2[i];
        }
    }
    
    void forward(Tensor<T> &output, const Tensor<T> &input, 
            const Tensor<bool> &tgt_mask)
    { 
        Tensor<T> tmp_in(input);
        int layer_num = 0;

        for (auto blockPtr: layers_gpt2) 
        {
            std::cout << "Forward pass of gpt2 decoder[" << layer_num++ << "]" << std::endl;

            blockPtr->forward(output, tmp_in, tgt_mask);
            tmp_in = output;
            std::cout << output << std::endl;
        }
    }

    uint64_t parameterCount() override 
    {
        uint64_t ret=0;

        for (int i = 0; i < layers_gpt2.size(); ++i) 
        {
            ret += layers_gpt2[i]->parameterCount();
        }
        return ret;
    }

private:
    std::vector<DecoderLayer_GPT2<T> *> layers_gpt2;
};
}


#endif //ATTENTION_TRANSFORMER_CPP_DECODER_GPT2_H
