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

#ifndef _MAX_TENSOR_H_
#define _MAX_TENSOR_H_

#include "layer.h"
#include "tensor.h"

using namespace std;
namespace text_attention
{
template<typename T>
class MaxTensor : virtual public Layer<T> 
{
public:
    void forward(Tensor<T>& output, const Tensor<T>& input) override
    {
        std::chrono::time_point<clock_> start_t = clock_::now();
        /* Extract shape information */
        int num_input = 1;
        int num_row = input.shape[0];
        int num_col = input.shape[1];
        uint64_t sz_outstack = 1;
        vector<int> out_shape;
        
        if (input.get_dims( )==3)
        {
            num_input = input.shape[0];
            num_row = input.shape[1];
            num_col = input.shape[2];
            out_shape = {num_input, num_row};
            sz_outstack = num_row;
        }
        else
            out_shape = {num_row, 1};

        /* Get max element from tensor */
        output.reshape(out_shape);
        for (int n=0; n<num_input; n++)
        {
            for (int i=0; i<num_row; i++)
            {
                uint64_t offset = n*num_row*num_col;
                int max_index = 
                    std::max_element(input.begin( )+offset+i*num_col,
                            input.begin( )+offset+(i+1)*num_col) 
                    - (input.begin( )+offset+i*num_col);
                
                output[n*sz_outstack+i] = max_index;
            }
        }
        interval_map["Max Tensor"] += INTERVAL(start_t);
    }

    uint64_t parameterCount() override 
    {
        return 0;
    }
};
};

#endif
