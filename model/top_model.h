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

#ifndef _TOP_MODEL_H_
#define _TOP_MODEL_H_

#include <bits/stdc++.h>
#include "tensor.h"

using namespace std;

namespace text_attention {
template<typename T>
class TopModel 
{
public:
    int num_layers;
    int dim_embed;
    int num_heads;
    int dim_ff;

    virtual void forward(Tensor<T>& output, const Tensor<T>& input) = 0;

    /*
     * Dimension will be [BATCH SIZE x Q_LEN x K LEN] 
     */
    void set_pad_mask(Tensor<bool>& mask, 
            const Tensor<T>& seq_q, const Tensor<T>& seq_k)
    {
        assert(seq_q.get_dims( )==2 && seq_k.get_dims( )==2);
        assert(seq_q.shape[0]==seq_k.shape[0]); // same batch size
        vector<int> mask_shape{seq_q.shape[0], seq_q.shape[1], seq_k.shape[1]};
        mask.reshape(mask_shape);

        uint64_t sz_stack = mask.shape[1]*mask.shape[2];
        for (int n=0; n<mask.shape[0]; n++)
        {
            for (int i=0; i<mask.shape[1]; i++)
            {
                for (int j=0; j<mask.shape[2]; j++)
                {
                    if (seq_k[n*mask.shape[2]+j]==2)
                        mask[n*sz_stack+i*mask.shape[2]+j] = false;
                    else
                        mask[n*sz_stack+i*mask.shape[2]+j] = true;
                }
            }
        }
    }

    void set_dec_mask(Tensor<bool>& mask, const Tensor<T>& input)
    {
        /* 
         * Note: dimension of 2 is to distinguish source & target masks 
         * Dimension will be [SENTENCE LEN SENTENCE LEN]
         * */
        vector<int> mask_shape{(int)input.size( ), (int)input.size( )};
        mask.reshape(mask_shape);
        for (int i=0; i<input.size( ); i++)
        {
            for (int j=0; j<input.size( ); j++)
            {
                if (i>=j)
                    mask[i*input.size( )+j] = true;
                else
                    mask[i*input.size( )+j] = false;
            }
        }
    }
};
};

#endif
