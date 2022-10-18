//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H
#define ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H

#include "top_model.h"
#include "layer.h"
#include "tensor.h"
#include "linear.h"
#include "functions.h"

using namespace std;

namespace text_attention {
template<typename T>
class FeedForward : virtual public Layer<T> {
public:
    FeedForward(TopModel<T>* master,
            int dim_model, int dim_ff, const string prefix_str, 
            const string weight_str,  const string bias_str, 
            const string ff_hidden_str, const string ff_out_str)
    : Layer<T>(master)
    {
        std::cout << ">>>> Init feedforward sublayer - " << std::endl;

        /* Get weight parameters */
        string prefix_hidden = prefix_str+"."+ff_hidden_str;
        string prefix_out = prefix_str+"."+ff_out_str;

        /* Tensorize */
        Tensor<T>* h_w = new Tensor<T>(
                param_map[prefix_hidden+"."+weight_str].pvals,
                param_map[prefix_hidden+"."+weight_str].pshape);
        Tensor<T>* h_b = new Tensor<T>(
                param_map[prefix_hidden+"."+bias_str].pvals,
                param_map[prefix_hidden+"."+bias_str].pshape);

        Tensor<T>* o_w = new Tensor<T>(
                param_map[prefix_out+"."+weight_str].pvals,
                param_map[prefix_out+"."+weight_str].pshape);
        Tensor<T>* o_b = new Tensor<T>(
                param_map[prefix_out+"."+bias_str].pvals,
                param_map[prefix_out+"."+bias_str].pshape);

        /* Init linear layers */
        linear_h = new Linear<T>(prefix_hidden, dim_model, dim_ff, *h_w, *h_b);
        linear_o = new Linear<T>(prefix_out, dim_ff, dim_model, *o_w, *o_b);

        linear_h->print_params( );
        linear_o->print_params( );
    }

    ~FeedForward() {
        delete linear_h;
        delete linear_o;
    }

    void forward(const Tensor <T> &input, Tensor <T> &output, 
            const Tensor<bool> &/*mask*/, const Tensor<T> &memory) override 
    {
        std::cout << "FFNN.Forward" << std::endl;

        Tensor<T> h2out{};
        linear_h->forward(input, h2out);

        for (int i=0; i<h2out.size( ); i++)
            h2out[i] = GELU(h2out[i]);

        linear_o->forward(h2out, output);
    }

    uint64_t parameterCount() override {
        return linear_h->parameterCount() + linear_o->parameterCount();
    }

private:
    Linear <T> *linear_h = nullptr;
    Linear <T> *linear_o = nullptr;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_FEED_FORWARD_H
