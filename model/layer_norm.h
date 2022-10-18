//
// Created by dianh on 2021/04/18.
//

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
            << " gamma.shape=" << gamma->size( ) 
            << " beta.shape=" << beta->size( ) << std::endl;
    }

    uint64_t parameterCount() override
    {
        return 0;
    }

    void forward(const Tensor <T> &input, Tensor <T> &output) override 
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
                    output[n*sz_stack+i*dim+j] = norm*(*gamma)[j]+(*beta)[j];
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

    T eps;
    std::vector<T> *gamma = nullptr; 
    std::vector<T> *beta = nullptr; 
    std::string name;
    int dim;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_LAYER_NORM_H
