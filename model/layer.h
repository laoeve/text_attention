//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_LAYER_H
#define ATTENTION_TRANSFORMER_CPP_LAYER_H

#include "top_model.h"
#include <bits/stdc++.h>
#include "tensor.h"

namespace text_attention {
template<typename T>
class Layer 
{
public:
    Layer( ) { }
    Layer(TopModel<T>* master) : master(master) { }
    virtual uint64_t parameterCount() = 0;
    virtual void forward(const Tensor<T> &/*input*/, Tensor<T> &/*output*/) 
    {
    } 
    virtual void forward(const Tensor<T> &/*input*/, Tensor<T> &/*output*/,
            const Tensor<bool> &/*mask*/, const Tensor<T> &/*memory*/) 
    {
    }

    virtual void print_params( ) { }

protected:
    TopModel<T>* master;
};

}
#endif //ATTENTION_TRANSFORMER_CPP_LAYER_H
