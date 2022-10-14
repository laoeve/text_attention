//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// based from stage_module.h
// 

#ifndef ATTENTION_TRANSFORMER_CPP_DECODER_H
#define ATTENTION_TRANSFORMER_CPP_DECODER_H

#include <vector>
#include "tensor.h"
#include "layer.h"
//#include "patch_merging.h"
#include "decoderLayer.h"

namespace text_attention {
    template<typename T>
    class Decoder : public Layer<T> {
    public:
        Decoder(int num_layer, int dim_model, int dim_ff, int heads, int max_len) {
            std::string str_key_layer = "decoder.layers.";
            for (int i = 0; i < num_layer; i += 1) {
                auto dec = new DecoderLayer<T>(num_layer, dim_model, dim_ff, heads, max_len, str_key_layer + std::to_string(i) + ".");
                layers.push_back(dec);
            }
        }

        ~Decoder() {
            for (int i = 0; i < layers.size(); ++i) {
                delete layers[i];
            }
        }
// changed from w-msa
        void forward(const Tensor<T> &input, Tensor<T> &output, Tensor<T> &memory, Tensor<T> &tgt_mask, Tensor<T> &src_mask) { 
            Tensor<T> tmp{};
            int layer_num = 0;
            tmp = input;
            for (auto blockPtr: layers) {
                std::cout << "Fwd Decoder." << layer_num++ << std::endl;
                Tensor<T> tmp_loop{};
                blockPtr->forward(tmp, memory, tmp_loop, tgt_mask, src_mask);
                tmp = tmp_loop;
            }
            output.clear();
            output.shape.clear();
            output.insert(output.end(), tmp.begin(), tmp.end());
            output.shape.insert(output.shape.end(), tmp.shape.begin(), tmp.shape.end());

            /*
            Tensor<T> tmp{};
            Tensor<T> tmp1{};
            Tensor<T> tmp2{};
            Tensor<T> tmp3{};
            Tensor<T> tmp4{};
            Tensor<T> tmp5{};
            Tensor<T> tmp6{};
            Tensor<T> tmp7{};
            enc1.forward(input, tmp1);
            enc2.forward(tmp1, tmp2);
            enc3.forward(tmp2, tmp3);
            enc4.forward(tmp3, tmp4);
            enc5.forward(tmp4, tmp5);
            enc6.forward(tmp5, tmp6);   
            layerNorm.forward(tmp6, tmp7);
            */

        }


        long long parameterCount() {
            long long ret=0;
            for (int i = 0; i < layers.size(); ++i) {
                ret += layers[i]->parameterCount();
            }
            return ret;
        }

    private:
        std::vector<DecoderLayer<T> *> layers;
    };
}


#endif //ATTENTION_TRANSFORMER_CPP_DECODER_H
