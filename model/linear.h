//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_LINEAR_H
#define ATTENTION_TRANSFORMER_CPP_LINEAR_H

#include "bits/stdc++.h"
#include "tensor.h"
#include "layer.h"

namespace text_attention {
template<typename T>
class Linear : public Layer<T> {
public:
    Linear(string name, int in_feature, int out_feature, 
            Tensor<T> &param_weights, Tensor<T> &param_bias) 
    : name(name), in_feature(in_feature), out_feature(out_feature) 
    {
        weights = &param_weights;
        bias = &param_bias;

        if (weights->shape[0]==out_feature && 
                weights->shape[1]==in_feature)
            weights->transpose( );
    }

    uint64_t parameterCount() override 
    {
        return weights->size() + bias->size();
    }

    ~Linear() {
        delete weights;
        delete bias;
    }

    void print_params( ) override 
    {
        std::cout << ">>>>>>>> Linear - " << name 
            << " weight.shape=" << *weights 
            << " bias.shape=" << *bias << std::endl;
    }

    void forward(Tensor <T> &output, const Tensor <T> &input) override 
    {
        if (is_operable(input)==false)
        {
            std::cerr << "Error: dimension error on " << name << std::endl;
            assert(0);
            exit(1);
        }

        /* Vector-matrix multiplication */
        multiply(input, output);
    }

private:
    //TODO: optimize in future if possible
    void multiply(const Tensor<T>& input, Tensor<T>& output)
    {
        /* Determine shapes of operators */
        int num_input = 1;
        int num_row = input.shape[0];
        vector<int> out_shape;
        if (input.get_dims( )==3)
        {
            num_input = input.shape[0];
            num_row = input.shape[1];
            out_shape.push_back(num_input);
            out_shape.push_back(num_row);
            out_shape.push_back(out_feature);
        }
        else
        {
            out_shape.push_back(num_row);
            out_shape.push_back(out_feature);
        }
        output.reshape(out_shape);
        
        uint64_t sz_instack = num_row*in_feature;
        uint64_t sz_outstack = num_row*out_feature;

        /* Multiply & add for each stack */
        for (int n=0; n<num_input; n++)
        {
            for (int i=0; i<num_row; i++)
            {
                for (int j=0; j<out_feature; j++)
                {
                    output[n*sz_outstack+i*out_feature+j] += (*bias)[j];
                    for (int k=0; k<in_feature; k++)
                    {
                        output[n*sz_outstack+i*out_feature+j] +=
                            input[n*sz_instack+i*in_feature+k]*
                            (*weights)[k*out_feature+j];
                    }
                }
            }
        }
    }

    bool is_operable(const Tensor<T>& op)
    {
        /* 
         * Dimension of '3' means multiple inputs 
         * op can be seen as [#INPUT x #WORDS x INPUT_FEATURE]
         * Each input is applied to matrix multiplication,
         * finally stack (=concatenate) all multiplied results
         * */
        uint64_t num_dims = op.get_dims( );
        if (num_dims>3 || num_dims==1 || num_dims==0)
            return false;

        if (op.shape[num_dims-1]!=in_feature)
            return false;

        return true;
    }

    Tensor<T> *weights = nullptr; 
    Tensor<T> *bias = nullptr;
    std::string name;
    int in_feature;
    int out_feature;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_LINEAR_H
