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

#ifndef _EMBEDDING_H_
#define _EMBEDDING_H_

#include "bits/stdc++.h"

#include "top_model.h"
#include "tensor.h"
#include "layer.h"

using namespace std;

namespace text_attention 
{
template<typename T>
class Embedding : public Layer<T>
{
public:
    Embedding(string name, int dim_model, Tensor<T>& lut_em, Tensor<T>& lut_pe)
    : name(name), dim_model(dim_model)
    {
        this->lut_em = &lut_em;
        this->lut_pe = &lut_pe;
    }

    uint64_t parameterCount( ) override
    {
        return (lut_em->size( )+lut_pe->size( ));
    }

    ~Embedding( )
    {
        delete lut_em;
        delete lut_pe;
    }

    void print_params( ) override
    {
        std::cout << "Init embedding " << name 
            << " embeddingTable.shape=" << *lut_em; 
        if(lut_pe->is_void( )==false)
            std::cout << " positionalEncoding.shape=" << *lut_pe << std::endl;
    }

    void forward(Tensor<T>& output, const Tensor<T>& input) override
    {
        /* Set shape */
        std::vector<int> out_shape = input.shape;
        out_shape.push_back(dim_model);
        output.reshape(out_shape);

        /* Set value */
        int num_input = output.shape[0];
        int len = output.shape[1];
        for (int n=0; n<num_input; n++)
        {
            for (int idx=0; idx<len; idx++)
            {
                for (int ebd=0; ebd<dim_model; ebd++)
                {
                    embed_pos = (*lut_em)[input[n*len+idx]*dim_model+ebd] *
                                std::sqrt(dim_model);
                    if (lut_pe->is_void( )==false)
                        embed_pos+= (*lut_pe)[idx*dim_model+ebd];
                    output[n*len*dim_model+idx*dim_model+ebd] = embed_pos;
                }
            }
        }
    }

private:
    Tensor<T>* lut_em; // embedding table
    Tensor<T>* lut_pe; // positional encoding table
    std::string name;
    int dim_model;
    T embed_pos;
};
};

#endif
