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

#ifndef ATTENTION_TRANSFORMER_CPP_ENCODER_H
#define ATTENTION_TRANSFORMER_CPP_ENCODER_H

#include "bits/stdc++.h"
#include "top_model.h"
#include "tensor.h"
#include "layer.h"
//#include "patch_merging.h"
#include "encoderLayer.h"

using namespace std;

namespace text_attention {
template<typename T>
class Encoder : public Layer<T> {
public:
    Encoder(TopModel<T>* master,
            int num_layers, int dim_model, int num_heads, int dim_ff,
            const string prefix_enc, const string prefix_layer,
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
            EncoderLayer<T>* enc = new EncoderLayer<T>(master,
                    dim_model, num_heads, dim_ff, prefix_enc, prefix_layer, id,
                    weight_str, bias_str, LN_gamma_str, LN_beta_str,
                    sa_query_str, sa_key_str, sa_value_str, sa_out_str,
                    ff_hidden_str, ff_out_str, LN_mh_str, LN_ff_str);
            layers.push_back(enc);
        }
    }

    ~Encoder() 
    {
        for (int i = 0; i < layers.size(); ++i) 
        {
            delete layers[i];
        }
    }

    void forward(Tensor<T> &output, const Tensor<T> &input, const Tensor<bool> &mask) { 
        Tensor<T> tmp_in(input);
        int layer_num = 0;
        for (auto blockPtr: layers) {
#ifdef DEBUG
            std::cout << "Forward pass of encoder[" << layer_num++ << "]" << std::endl;
#endif
            layer_num = layer_num;

            blockPtr->forward(output, tmp_in, mask);
            tmp_in = output;
        }
    }

    uint64_t parameterCount() override 
    {
        uint64_t ret=0;
        for (int i = 0; i < layers.size(); ++i) 
        {
            ret += layers[i]->parameterCount();
        }
        return ret;
    }

private:
    std::vector<EncoderLayer<T> *> layers;
};
}


#endif //ATTENTION_TRANSFORMER_CPP_ENCODER_H
