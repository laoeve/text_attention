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

    //TODO: optimize in future if possible
    void matmul(Tensor<T>& out, const Tensor<T>& opa, 
            const Tensor<T>& opb, const T scale_factor)
    {
        assert(opa.shape[1]==opb.shape[0]);
        
        /* Determine output shapes */
        int num_row = opa.shape[0];
        int num_col = opb.shape[1];
        int num_col_opa = opa.shape[1];
        vector<int> out_shape{num_row, num_col};
        out.reshape(out_shape);

        /* Matrix multiplication */
        for (int i=0; i<num_row; i++)
        {
            for (int j=0; j<num_col; j++)
            {
                out[i*num_col+j] = 0;
                for (int k=0;k<num_col_opa; k++)
                {
                    out[i*num_col+j] += 
                        opa[i*num_col_opa+k]*opb[k*num_col+j];
                }
                out[i*num_col+j] *= scale_factor;
            }
        }
    }

protected:
    TopModel<T>* master;
};

}
#endif //ATTENTION_TRANSFORMER_CPP_LAYER_H
