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

    void forward(const Tensor <T> &input, Tensor <T> &output) override {
        assert(input.shape[input.shape.size() - 1] == dim);
        output.shape.clear();
        output.shape.insert(output.shape.end(), input.shape.begin(), input.shape.end());
        for (int i = 0; i < input.size(); i += dim) {
            // Var(x) = E(x^2) - E(x)^2
            T avg = 0;
            T avg2 = 0;
            T sum = 0;
            T sum2 = 0;
            for (int j = 0; j < dim; ++j) {
                sum += input[i + j];
                sum2 += input[i + j] * input[i + j];
            }
            avg = sum / dim, avg2 = sum2 / dim;
            T varx = avg2 - avg * avg;

            // y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta
            for (int j = 0; j < dim; ++j) {
                output.push_back((input[i + j] - avg) / sqrt(varx + eps) * (double)gamma->at(j) + (double)beta->at(j));
            }
        }
    }

private:
    T eps;
    std::vector<T> *gamma = nullptr; 
    std::vector<T> *beta = nullptr; 
    std::string name;
    int dim;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_LAYER_NORM_H
