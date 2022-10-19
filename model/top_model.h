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

    virtual void forward(const Tensor<T> &input, Tensor<T> &output) = 0;

    void set_enc_mask(const Tensor<T>& input, Tensor<bool>& mask) 
    {
        /*
         * Dimension will be [SENTENCE NUM x SENTENCE LEN] 
         */
        assert(input.shape.size( )==2);
        int num_words = input.shape[1];
        mask.reshape(vector<int>{input.shape[0], input.shape[1]});
        for (int i=0; i<input.shape[0]; i++)
        {
            for (int j=0; j<input.shape[1]; j++)
            {
                if (input[i*num_words+j]==2)
                    mask[i*num_words+j] = false;
                else
                    mask[i*num_words+j] = true;
            }
        }
    }

    void set_dec_mask(const Tensor<T>& input, Tensor<bool>& mask)
    {
        /* 
         * Note: dimension of 3 is to distinguish source & target masks 
         * Dimension will be [1 SENTENCE LEN SENTENCE LEN]
         * */
        vector<int> mask_shape{1, (int)input.size( ), (int)input.size( )};
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
