//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// swin_transformer.h

#ifndef ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H
#define ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H

#include <bits/stdc++.h>

#include "layer.h"
#include "tensor.h"
#include "encoder.h"
#include "decoder.h"
#include "layer_norm.h"
#include "softmax.h"
#include "linear.h"

namespace text_attention {
    template<typename T>
    class AttentionTransformer : virtual public Layer<T> {
    public:
        AttentionTransformer(int num_layer, int dim_model, int dim_ff, int heads,
                            int src_vocab, int tgt_vocab, int max_len) : src_vocab(src_vocab), dim_model(dim_model), max_len(max_len){
            encoder = new Encoder<T>(num_layer, dim_model, dim_ff, heads, max_len);
            decoder = new Decoder<T>(num_layer, dim_model, dim_ff, heads, max_len);
            
            // generator.proj.weight.size[6384,512] generator.proj.bias.Size([6384])
            auto gen_w = new Tensor<T>{};
            gen_w->insert(gen_w->end(), param_map["generator.proj.weight"].pvals.begin(), param_map["generator.proj.weight"].pvals.end());
            gen_w->shape.insert(gen_w->shape.end(), param_map["generator.proj.weight"].pshape.begin(), param_map["generator.proj.weight"].pshape.end());
            t_tran(*gen_w);
            auto gen_b = new Tensor<T>{};
            gen_b->insert(gen_b->end(), param_map["generator.proj.bias"].pvals.begin(), param_map["generator.proj.bias"].pvals.end());
            gen_b->shape.insert(gen_b->shape.end(), param_map["generator.proj.bias"].pshape.begin(), param_map["generator.proj.bias"].pshape.end());

            auto generator = new Linear<T>(dim_model, tgt_vocab, *gen_w, *gen_b);
        }


        AttentionTransformer(int src_vocab, int tgt_vocab)
                : AttentionTransformer(6, 512, 2048, 8, src_vocab, tgt_vocab, text_attention::max_len) {
        }

        void forward(Tensor<T> &input, Tensor<T> &output) {
            Tensor<T> memory{};
            Tensor<T> src_mask{};
            Tensor<T> input_embed{};
            std::vector<float> lut;

            // making encoder mask : tensor( x for x in input != pad)
            for(int i = 0; i < input.shape[1] ; i++){
                if(input[(1*i)] == 2){
                    src_mask.push_back(false);
                } else {
                    src_mask.push_back(true);
                }
            }
            src_mask.shape = {1, 1, input.shape[1]};

            embed_idx(input, input_embed, dim_model, "src_embed.0.lut.weight", "src_embed.1.pe");

            encoder->forward(input_embed, memory, src_mask);
            
            // encoder input : tensor([[[0]]]).embed import
            Tensor<T> tgt_input{};
            Tensor<T> tgt_embed{};
            Tensor<T> tgt_mask{};
            Tensor<T> tmp3{};
            Tensor<T> decoder_out{};

            tgt_input.push_back(0);
            tgt_input.shape={1,1};

            // decoder mask : [1,0][1,1] ...
            // encoder mask : same with encoder mask used in encoder
            // Decoder loop for expect sentence
            for(int i=0; i < this->max_len; i++){
                //making tgt_mask.shape={j,j};
                tgt_mask.clear();
                for(int j = 1; j < tgt_input.size() ; j++){
                    tgt_mask.push_back(true);
                    if(j != 1){
                        for(int m_size = 0; m_size < j; m_size++){      // [1,1,..,1,0, ..,0,0]
                            std::vector<int> m_front(m_size,1); // [1,1 ...]
                            std::vector<int> m_back(j-m_size-1,0);  // [0,0, ...]
                            m_front.insert(m_front.end(), m_back.begin(), m_back.end());    // [1, .. ,1,0, .. ,0]
                            tgt_mask.insert(tgt_mask.end(), m_front.begin(), m_front.end());
                        }
                    }
                }
                tgt_mask.shape.clear();
                tgt_mask.shape = {tgt_input.size(), tgt_input.size()};

                embed_idx(tgt_input, tgt_embed, dim_model, "tgt_embed.0.lut.weight", "tgt_embed.1.pe");

                decoder->forward(tgt_embed, decoder_out, memory, tgt_mask, src_mask);  
                generator->forward(decoder_out, tmp3);
                softMax.forward(tmp3, output);

                //find max value of probability
                int max_index = max_element(output.begin(), output.end()) - output.begin();
                std::cout << "next word : " << max_index << std::endl;
                
                tgt_input.push_back(max_index);
                tgt_input.shape = {1,tgt_input.shape[1] + 1};
            }
        }

        long long parameterCount() {
            return encoder->parameterCount() + decoder->parameterCount() +
                   generator->parameterCount() + softMax.parameterCount();
        }

    private:
        int max_len;
        int src_vocab;
        int dim_model;
        Encoder<T> *encoder = nullptr;
        Decoder<T> *decoder = nullptr;
        Linear<T> *generator = nullptr;
        SoftMax<T> softMax;
        
    };
}

#endif //ATTENTION_TRANSFORMER_CPP_ATTENTION_TRANSFORMER_H
