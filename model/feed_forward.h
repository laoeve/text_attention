//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H
#define ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H

#include "layer.h"
#include "tensor.h"
#include "linear.h"
#include "functions.h"

namespace text_attention {
    template<typename T>
    class FeedForward : virtual public Layer<T> {
    public:
        FeedForward(int dim_model, int dim_ff, std::string str_key_layer) : dim_model(dim_model), dim_ff(dim_ff) {
            std::cout << "Init.FeedForward" << " " << str_key_layer <<std::endl;
            auto w1_w = new Tensor<T>{};
            auto w1_b = new Tensor<T>{};
            auto w2_w = new Tensor<T>{};
            auto w2_b = new Tensor<T>{};;
            w1_w->insert(w1_w->end(), text_attention::param_map[str_key_layer+"w_1.weight"].pvals.begin(), text_attention::param_map[str_key_layer+"w_1.weight"].pvals.end());
            w1_w->shape.insert(w1_w->shape.end(), text_attention::param_map[str_key_layer+"w_1.weight"].pshape.begin(), text_attention::param_map[str_key_layer+"w_1.weight"].pshape.end());
            t_tran(*w1_w);
            w1_b->insert(w1_b->end(), text_attention::param_map[str_key_layer+"w_1.bias"].pvals.begin(), text_attention::param_map[str_key_layer+"w_1.bias"].pvals.end());
            w1_b->shape.insert(w1_b->shape.end(), text_attention::param_map[str_key_layer+"w_1.bias"].pshape.begin(), text_attention::param_map[str_key_layer+"w_1.bias"].pshape.end());
            w2_w->insert(w2_w->end(), text_attention::param_map[str_key_layer+"w_2.weight"].pvals.begin(), text_attention::param_map[str_key_layer+"w_2.weight"].pvals.end());
            w2_w->shape.insert(w2_w->shape.end(), text_attention::param_map[str_key_layer+"w_2.weight"].pshape.begin(), text_attention::param_map[str_key_layer+"w_2.weight"].pshape.end());
            t_tran(*w2_w);
            w2_b->insert(w2_b->end(), text_attention::param_map[str_key_layer+"w_2.bias"].pvals.begin(), text_attention::param_map[str_key_layer+"w_2.bias"].pvals.end());
            w2_b->shape.insert(w2_b->shape.end(), text_attention::param_map[str_key_layer+"w_2.bias"].pshape.begin(), text_attention::param_map[str_key_layer+"w_2.bias"].pshape.end());
            linear1 = new Linear<T>(dim_model, dim_ff, *w1_w, *w1_b);
            linear2 = new Linear<T>(dim_ff, dim_model, *w2_w, *w2_b);
        }

        ~FeedForward() {
            if (linear1 != nullptr) {
                delete linear1;
                linear1 = nullptr;
            }
            if (linear2 != nullptr) {
                delete linear2;
                linear2 = nullptr;
            }
        }

        void forward(const Tensor <T> &input, Tensor <T> &output, Tensor<T> &mask, Tensor<T> &memory) {
            std::cout << "FFNN.Forward" << std::endl;
            Tensor<T> tmp1{};
            linear1->forward(input, tmp1);
            std::cout << tmp1 << std::endl;
            Tensor<T> tmp2{};
            tmp2.shape.insert(tmp2.shape.end(), tmp1.shape.begin(), tmp1.shape.end());
            for (auto item: tmp1) {
                tmp2.push_back(GELU(item));
            }
            linear2->forward(tmp2, output);
            std::cout<<output <<std::endl;

        }

        long long parameterCount() {
            return linear1->parameterCount() + linear2->parameterCount();
        }

    private:
        Linear <T> *linear1 = nullptr;
        Linear <T> *linear2 = nullptr;
        int dim_model, dim_ff;
    };
}
#endif //ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H
