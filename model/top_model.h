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
};
};

#endif
