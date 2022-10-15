//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// we changed the linear function to import the weight from external files

#ifndef ATTENTION_TRANSFORMER_CPP_LINEAR_H
#define ATTENTION_TRANSFORMER_CPP_LINEAR_H

#include "tensor.h"
#include "layer.h"

namespace text_attention {
    template<typename T>
    class Linear : public Layer<T> {
    public:

        Linear(int in_feature, int out_feature, Tensor<T> &param_weights, Tensor<T> &param_bias) : in_feature(in_feature),
                                                 out_feature(out_feature) {
            weights = &param_weights;
            bias = &param_bias;
        }

        long long parameterCount() {
            if (true)   //use_bias
                return weights->size() + bias->size();
            return weights->size();
        }

        ~Linear() {
            if (weights != nullptr) {
                delete weights;
                weights = nullptr;
            }
            if (bias != nullptr) {
                delete bias;
                bias = nullptr;
            }
        }

/**
 *
 * @param input DIM x INPUT_FEATURE     //d_model   512
 * @param output DIM X OUTPUT_FEATURE   //d_k   64
 */
        void forward(const Tensor <T> &input, Tensor <T> &output) {
            assert(input.shape.size() and input.shape.back() == this->in_feature);
            output.shape.clear();
            output.shape.insert(output.shape.end(), input.shape.begin(), input.shape.end());    //input{1,128,512}
            output.shape.back() = out_feature;  //output{1,128,64}
            std::vector<T> tmp{};

            for (auto pos = 0; pos < input.size(); pos += in_feature) {
                if (true) { 
                    for(int idx_a = 0 ; idx_a < out_feature ; idx_a ++){
                        tmp.push_back(bias->at(idx_a));
                    }
                }

                for (int j = 0; j < out_feature; ++j) {
                    for (int i = 0; i < in_feature; ++i) {
                        tmp[j] += input[pos + i] * weights->at(j + i * out_feature);
                    }   //weight{1,512,64}
                }
                output.insert(output.end(), tmp.begin(), tmp.end());
                tmp.clear();
            }
/* 
            std::cout << "for bias : " << b << std::endl;
            std::cout << "for matmul : " << a << std::endl;

            std::cout << "Linear input   : " << input << std::endl;
            std::cout << "Linear output  : " << output << std::endl; */
            assert(output.size() == ((input.size() / in_feature) * out_feature));
        }

    private:
        Tensor<T> *weights = nullptr; 
        Tensor<T> *bias = nullptr;
        int in_feature, out_feature;
    };
}
#endif //ATTENTION_TRANSFORMER_CPP_LINEAR_H
