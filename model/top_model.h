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
        assert(input.shape.size( )==2);
        int num_words = input.shape[1];
        mask.reshape(vector<int>{1, input.shape[0], input.shape[1]});
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
        vector<int> mask_shape{(int)input.size( ), (int)input.size( )};
        mask.reshape(mask_shape);
        mask.clear( ); // TODO: unsafe usage -> FIXME
        for (int i=1; i<input.size( ); i++)
        {
            if (i==1) continue;

            for (int mask_sz=0; mask_sz<i; mask_sz++)
            {
                vector<bool> mask_front(mask_sz, true);
                vector<bool> mask_back(i-mask_sz-1, false);
                mask_front.insert(mask_front.end( ), 
                        mask_back.begin( ), mask_back.end( ));
                mask.insert(mask.end( ), mask_front.begin( ), mask_front.end( ));
            }
        }
    }
};
};

#endif
