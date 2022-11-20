//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_LAYER_H
#define ATTENTION_TRANSFORMER_CPP_LAYER_H

#include "bits/stdc++.h"
#include "top_model.h"
#include "tensor.h"

namespace text_attention {
template<typename T>
class Layer 
{
public:
    Layer( ) { }
    Layer(TopModel<T>* master) : master(master) { }
    virtual uint64_t parameterCount() = 0;
    virtual void forward(Tensor<T> &/*output*/, const Tensor<T> &/*input*/) 
    {
    } 
    virtual void forward(Tensor<T> &/*output*/, const Tensor<T> &/*input*/, 
            const Tensor<bool> &/*mask*/, const Tensor<T> &/*memory*/) 
    {
    }

    virtual void print_params( ) { }

    //TODO: optimize in future if possible
    void matmul(Tensor<T>& out, const Tensor<T>& opa, 
            const Tensor<T>& opb, const T scale_factor)
    {
        std::chrono::time_point<clock_> start_t = clock_::now();
#ifdef DEBUG
        if ((opa.get_dims( )==1 && opb.get_dims( )==1 && opb.shape[0]!=1) ||
            (opa.get_dims( )==2 && opa.shape[1]!=opb.shape[0]))
        {
            std::cerr << "Error: dimension error at "
                << "matrix multiplication" << std::endl;
            assert(0);
            exit(1);
        }
#endif
        
        /* Determine output shapes */
        int num_row = opa.shape[0];
        int num_col = (opb.get_dims( )==1)? 1 : opb.shape[1];
        int num_col_opa = (opa.get_dims( )==1)? 1 : opa.shape[1];
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
            interval_map["layer_matmul"] += INTERVAL(start_t);
}

protected:
    TopModel<T>* master;
};

}
#endif //ATTENTION_TRANSFORMER_CPP_LAYER_H