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

#ifndef ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H
#define ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H

#include "bits/stdc++.h"
#include "top_model.h"
#include "functions.h"
#include "layer.h"
#include "tensor.h"
#include "linear.h"

using namespace std;

namespace text_attention {
template<typename T>
class FeedForward : virtual public Layer<T> {
public:
    FeedForward(TopModel<T>* master,
            int dim_model, int dim_ff, const string prefix_str, 
            const string weight_str,  const string bias_str, 
            const string ff_hidden_str, const string ff_out_str)
    : Layer<T>(master)
    {
        std::cout << ">>>> Init feedForward sublayer - " << std::endl;

        /* Get weight parameters */
        string prefix_hidden = prefix_str+"."+ff_hidden_str;
        string prefix_out = prefix_str+"."+ff_out_str;

        /* Tensorize */
        Tensor<T>* h_w = new Tensor<T>(
                param_map[prefix_hidden+"."+weight_str].pvals,
                param_map[prefix_hidden+"."+weight_str].pshape);

        Tensor<T>* o_w = new Tensor<T>(
                param_map[prefix_out+"."+weight_str].pvals,
                param_map[prefix_out+"."+weight_str].pshape);

        Tensor<T>* h_b = nullptr;
        Tensor<T>* o_b = nullptr;
        if (bias_str.empty( ))
        {
            Tensor<T>* h_b = new Tensor<T> { };
            Tensor<T>* o_b = new Tensor<T> { };
        }
        else
        {
            Tensor<T>* h_b = new Tensor<T>(
                param_map[prefix_hidden+"."+bias_str].pvals,
                param_map[prefix_hidden+"."+bias_str].pshape);

            Tensor<T>* o_b = new Tensor<T>(
                param_map[prefix_out+"."+bias_str].pvals,
                param_map[prefix_out+"."+bias_str].pshape);
        }

        /* Init linear layers */
        linear_h = new Linear<T>(prefix_hidden, dim_model, dim_ff, *h_w, *h_b);
        linear_o = new Linear<T>(prefix_out, dim_ff, dim_model, *o_w, *o_b);

        linear_h->print_params( );
        linear_o->print_params( );
    }

    ~FeedForward() {
        delete linear_h;
        delete linear_o;
    }

    void forward(Tensor <T> &output, const Tensor <T> &input, 
            const Tensor<bool> &/*mask*/, const Tensor<T> &memory) override 
    {
        Tensor<T> h2out{};
        linear_o->forward(output, h2out);

        for (int i=0; i<h2out.size( ); i++)
            h2out[i] = GELU(h2out[i]);

        linear_o->forward(h2out, output);
    }

    uint64_t parameterCount() override {
        return linear_h->parameterCount() + linear_o->parameterCount();
    }

private:
    Linear <T> *linear_h = nullptr;
    Linear <T> *linear_o = nullptr;

};
}
#endif //ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H
