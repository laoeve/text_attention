//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// based from stage_module.h
// 

#ifndef ATTENTION_TRANSFORMER_CPP_ENCODER_H
#define ATTENTION_TRANSFORMER_CPP_ENCODER_H

#include <vector>
#include "tensor.h"
#include "layer.h"
#include "encoderLayer.h"

namespace text_attention {
    template<typename T>
    class Encoder : public Layer<T> {
    public:
        Encoder(int num_layer, int dim_model, int dim_ff, int heads, int max_len) {
            std::string str_key_layer = "encoder.layers.";
            for (int i = 0; i < num_layer; i += 1) {
                auto enc = new EncoderLayer<T>(num_layer, dim_model, dim_ff, heads, max_len, str_key_layer + std::to_string(i) + ".");
                layers.push_back(enc);
            }
            layerNorm = new LayerNorm<T>(dim_model, text_attention::param_map["encoder.norm.a_2"].pvals, text_attention::param_map["encoder.norm.a_2"].pvals);
        }

        ~Encoder() {
            for (int i = 0; i < layers.size(); ++i) {
                delete layers[i];
            }
            if (layerNorm != nullptr) {
                delete layerNorm;
                layerNorm = nullptr;
            }   
        }
// changed from w-msa
        void forward(const Tensor<T> &input, Tensor<T> &output, Tensor<T> &mask) { 
            
            Tensor<T> tmp{};
            int layer_num = 0;
            tmp = input;
            for (auto blockPtr: layers) {
                std::cout << "Fwd Encoder." << layer_num++ << std::endl;
                Tensor<T> tmp_loop{};
                blockPtr->forward(tmp, tmp_loop, mask);
                tmp = tmp_loop;
            }
            layerNorm->forward(tmp, output);

        }


        long long parameterCount() {
            long long ret=0;
            for (int i = 0; i < layers.size(); ++i) {
                ret += layers[i]->parameterCount();
            }
            return ret;
        }

    private:
        std::vector<EncoderLayer<T> *> layers;
        LayerNorm <T> *layerNorm = nullptr;
    };
}


#endif //ATTENTION_TRANSFORMER_CPP_ENCODER_H
