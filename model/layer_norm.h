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

#ifndef ATTENTION_TRANSFORMER_CPP_LAYER_NORM_H
#define ATTENTION_TRANSFORMER_CPP_LAYER_NORM_H

#include "layer.h"

using namespace std;

namespace text_attention {
template<typename T>
class LayerNorm : public Layer<T> {
public:
    LayerNorm(string name, int dim, vector<T> &vec_gamma, vector<T> &vec_beta)
    : name(name), dim(dim) 
    {
        eps = 1e-5;
        gamma = &vec_gamma;
        beta = &vec_beta;
    }

    ~LayerNorm( )
    {
        delete gamma;
        delete beta;
    }

    void print_params( ) override
    {
        std::cout << ">>>>>>>> LayerNorm - " << name 
            << " gamma.shape=" << gamma->size( );
        if(beta->empty( ) == false)
            std::cout << " beta.shape=" << beta->size( ) << std::endl;
    }

    uint64_t parameterCount() override
    {
        return 0;
    }

    void forward(Tensor <T> &output, const Tensor <T> &input) override 
    {
        if (is_operable(input)==false)
        {
            std::cerr << "Error: dimension error on " << name << std::endl;
            assert(0);
            exit(1);
        }

        /* Determine shapes of operators */
        int num_input = 1;
        int num_row = input.shape[0];
        if (input.get_dims( )==3)
        {
            num_input = input.shape[0];
            num_row = input.shape[1];
        }
        else if (input.get_dims( )==1)
            num_row = 1;
        output.reshape(input.shape);

        uint64_t sz_stack = num_row*dim;

        /* Layer normalization */
        for (int n=0; n<num_input; n++)
        {
            for (int i=0; i<num_row; i++)
            {
                /* Mean */
                float mean_x = 0.;
                for (int j=0; j<dim; j++)
                    mean_x += input[n*sz_stack+i*dim+j];
                mean_x /= dim;

                /* Variance */
                float var_x = 0.;
                for (int j=0; j<dim; j++)
                    var_x += (input[n*sz_stack+i*dim+j]-mean_x)*
                        (input[n*sz_stack+i*dim+j]-mean_x);
                var_x /= dim;

                /* Normalization */
                float denominator = sqrt(var_x+eps);
                for (int j=0; j<dim; j++)
                {
                    float norm = (input[n*sz_stack+i*dim+j]-mean_x)/denominator;
                    value_out = norm*((*gamma)[j]);
                    if (beta->empty( )==false)
                        value_out += ((*beta)[j]);
                    output[n*sz_stack+i*dim+j] = value_out;
                }
            }
        }
    }

private:
    bool is_operable(const Tensor<T>& op)
    {
        uint64_t num_dims = op.get_dims( );
        if (num_dims>3 || num_dims==0)
            return false;

        if (op.shape[num_dims-1]!=dim)
            return false;

        return true;
    }

    float eps;
    T value_out;
    std::vector<T> *gamma = nullptr; 
    std::vector<T> *beta = nullptr; 
    std::string name;
    int dim;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_LAYER_NORM_H
