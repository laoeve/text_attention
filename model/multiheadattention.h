//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// based from window_attention.h

#ifndef ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H
#define ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H

#include <cmath>

#include "top_model.h"
#include "layer.h"
#include "tensor.h"
#include "functions.h"
#include "linear.h"
#include "softmax.h"

using namespace std;

namespace text_attention {
template<typename T>
class MultiheadAttention : virtual public Layer<T> {
public:
    MultiheadAttention(TopModel<T>* master,
            int dim_model, int num_heads, const string prefix_str,
            const string weight_str, const string bias_str, 
            const string sa_query_str, const string sa_key_str,
            const string sa_value_str, const string sa_out_str)
    : Layer<T>(master), dim_model(dim_model), scale(1/sqrt(dim_model/heads)),
      heads(num_heads), headDim(dim_model/num_heads)
    {
        std::cout << ">>>> Init multihead sublayer - " << std::endl;
        uint64_t sanity_cntr = 0;

        /* Setup indexing string for query, key, value, output parameters */
        string wq_str = prefix_str+"."+sa_query_str+"."+weight_str;
        string wk_str = prefix_str+"."+sa_key_str+"."+weight_str;
        string wv_str = prefix_str+"."+sa_value_str+"."+weight_str;
        string bq_str = prefix_str+"."+sa_query_str+"."+bias_str;
        string bk_str = prefix_str+"."+sa_key_str+"."+bias_str;
        string bv_str = prefix_str+"."+sa_value_str+"."+bias_str;
        string wout_str = prefix_str+"."+sa_out_str+"."+weight_str;
        string bout_str = prefix_str+"."+sa_out_str+"."+bias_str;

        /* 'Tensorize' */
        Tensor<T> in_Wq(param_map[wq_str].pvals, param_map[wq_str].pshape);
        Tensor<T> in_Wk(param_map[wk_str].pvals, param_map[wk_str].pshape);
        Tensor<T> in_Wv(param_map[wv_str].pvals, param_map[wv_str].pshape);
        Tensor<T> in_Bq(param_map[bq_str].pvals, param_map[bq_str].pshape);
        Tensor<T> in_Bk(param_map[bk_str].pvals, param_map[bk_str].pshape);
        Tensor<T> in_Bv(param_map[bv_str].pvals, param_map[bv_str].pshape);
        Tensor<T>* w_out = 
            new Tensor<T>(param_map[wout_str].pvals, param_map[wout_str].pshape);
        Tensor<T>* b_out = 
            new Tensor<T>(param_map[bout_str].pvals, param_map[bout_str].pshape);

        /* Divide parametes into multiple heads */
        std::vector<Tensor<T> *>w_q;
        std::vector<Tensor<T> *>w_k;
        std::vector<Tensor<T> *>w_v;
        std::vector<Tensor<T> *>b_q;
        std::vector<Tensor<T> *>b_k;
        std::vector<Tensor<T> *>b_v;
        for (int h=0; h<heads; h++)
        {
            /* Weight division */
            Tensor<T>* tmp_Wq = new Tensor<T>{};
            Tensor<T>* tmp_Wk = new Tensor<T>{};
            Tensor<T>* tmp_Wv = new Tensor<T>{};

            for (int line=0; line<dim_model; line++)
            {
                for (int d_k=0; d_k<headDim; d_k++)
                {
                    sanity_cntr++;
                    tmp_Wq->push_back(in_Wq[line*dim_model+h*headDim+d_k]);
                    tmp_Wk->push_back(in_Wk[line*dim_model+h*headDim+d_k]);
                    tmp_Wv->push_back(in_Wv[line*dim_model+h*headDim+d_k]);
                }
            }
            assert(sanity_cntr==dim_model * headDim);

            tmp_Wq->shape = {dim_model, headDim};
            tmp_Wk->shape = {dim_model, headDim};
            tmp_Wv->shape = {dim_model, headDim};

            w_q.push_back(tmp_Wq);
            w_k.push_back(tmp_Wk);
            w_v.push_back(tmp_Wv);

            /* Bias division */
            Tensor<T>* tmp_Bq = new Tensor<T>{};
            Tensor<T>* tmp_Bk = new Tensor<T>{};
            Tensor<T>* tmp_Bv = new Tensor<T>{};

            sanity_cntr =0;
            for(int d_k = 0; d_k < headDim; ++d_k){
                sanity_cntr++;
                tmp_Bq->push_back(in_Bq[h*headDim+d_k]);
                tmp_Bk->push_back(in_Bq[h*headDim+d_k]);
                tmp_Bv->push_back(in_Bq[h*headDim+d_k]);
            }
            assert(sanity_cntr==headDim);

            tmp_Bq->shape = {headDim};
            tmp_Bk->shape = {headDim};
            tmp_Bv->shape = {headDim};

            b_q.push_back(tmp_Bq);
            b_k.push_back(tmp_Bk);
            b_v.push_back(tmp_Bv);
            
            sanity_cntr = 0;
        }

        /* Init linear layers */
        for (int h=0; h<heads; h++)
        {
            Linear<T>* lin_q = new Linear<T>(
                    prefix_str+"."+sa_query_str+".HD["+to_string(h)+"]", 
                    dim_model, headDim, *w_q[h], *b_q[h]);
            Linear<T>* lin_k = new Linear<T>(
                    prefix_str+"."+sa_key_str+".HD["+to_string(h)+"]", 
                    dim_model, headDim, *w_k[h], *b_k[h]);
            Linear<T>* lin_v = new Linear<T>(
                    prefix_str+"."+sa_value_str+".HD["+to_string(h)+"]", 
                    dim_model, headDim, *w_v[h], *b_v[h]);

            qLinear.push_back(lin_q);   //{1,128,64}
            kLinear.push_back(lin_k);
            vLinear.push_back(lin_v);

            lin_q->print_params( );
            lin_k->print_params( );
            lin_v->print_params( );
        }
        
        outLinear = new Linear<T>(prefix_str+"."+sa_out_str,
                dim_model, dim_model, *w_out, *b_out);
        outLinear->print_params( );
    }

    void forward(const Tensor<T> &input, Tensor<T> &output, 
            const Tensor<bool> &mask, const Tensor<T> &memory) override 
    {
        std::cout << "MultiheadAttention.Forward" << std::endl;
        //int dim_k = this->dim_model / this->heads;
        Tensor<T> tmp1{};

        // lin(x) for lin, x in zip(self.linears, (query, key, value))
        tmp1.insert(tmp1.end(), input.begin(), input.end());
        tmp1.shape.insert(tmp1.shape.end(), input.shape.begin(), input.shape.end());    //{1,128,512}
        std::cout << tmp1 << std::endl;

        if(&memory != nullptr){
            for(auto item : (this->qLinear)){
                auto *q_tmp = new Tensor<T>{};
                item->forward(memory, *q_tmp);
                q.push_back(q_tmp);
            }
            for(auto item : (this->kLinear)){
                auto *k_tmp = new Tensor<T>{};
                item->forward(memory, *k_tmp);
                k.push_back(k_tmp);
            }
            for(auto item : (this->vLinear)){
                auto *v_tmp = new Tensor<T>{};
                item->forward(tmp1, *v_tmp);
                v.push_back(v_tmp);
            }
        } else {
            for(auto item : (this->qLinear)){
                auto *q_tmp = new Tensor<T>{};
                item->forward(tmp1, *q_tmp);
                q.push_back(q_tmp);
            }
            for(auto item : (this->kLinear)){
                auto *k_tmp = new Tensor<T>{};
                item->forward(tmp1, *k_tmp);
                k.push_back(k_tmp);
            }
            for(auto item : (this->vLinear)){
                auto *v_tmp = new Tensor<T>{};
                item->forward(tmp1, *v_tmp);
                v.push_back(v_tmp);
            }
        }
        tmp1.clear();
        tmp1.shape.clear();

        Tensor<T> dots{};
        Tensor<T> tmp2{};
        Tensor<T> tmp3{};

        // 2) scale dot attention
        // matmul(q, k^t)
        for (int h = 0; h < heads; ++h) {
            int num = q[h]->shape[q[h]->shape.size() - 2];    // 128    len_word
            int d_k = q[h]->shape[q[h]->shape.size() - 1];    // 64     d_embed == d_k
            for (int pi = 0; pi < num; ++pi) {
                for (int pj = 0; pj < num; ++pj) {
                    T val = 0;
                    for (int pk = 0; pk < d_k; ++pk) {
                        val += (*q[h])[pi * d_k + pk] * (*k[h])[pj * d_k + pk] * scale; /* added scale product */
                    }
                    dots.push_back(val);
                }
            }
            dots.shape.clear();
            dots.shape.insert(dots.shape.begin(), k[h]->shape.begin(), k[h]->shape.end());   // {1,128,64}
            dots.shape[dots.shape.size() - 1] = (*k[h]).shape[(*k[h]).shape.size() - 2];  //  scores{1,128,128}
/* 
            std::cout << "dots shape : " << dots.shape[0] << " " << dots.shape[1] << " " << dots.shape[2] << std::endl;
            std::cout << "mask shape : " << mask.shape[0] << " " << mask.shape[1] << " " << mask.shape[2] << std::endl;
            std::cout << "verify dots Tensor : " << dots << std::endl;
*/
            if (&mask != nullptr) {
                if(mask.shape[mask.shape.size()-1] == mask.shape[mask.shape.size()-2]){     //condition : src_mask {num * 128}
                    for (int p = 0; p < mask.shape[1]; ++p) {   // count num
                        for (int j = 0; j < mask.shape[2]; ++j){   //count 128
                            if(mask[j] == false){
                                dots[mask.shape[2] * p + j] = -1e9;
                            }
                        }
                    }
                } else {      //condition : tgt_mask {num * num}
                    for (int p = 0; p < mask.shape[1]; ++p) {   // count num
                        for (int j = 0; j < mask.shape[2]; ++j){   //count 128
                            if(mask[mask.shape[2] * p + j] == false){
                                dots[mask.shape[2] * p + j] = -1e9;
                            }
                        }
                    }                        
                }
            }

            Tensor<T> attn{};
            softMax.forward(dots, attn);    //softmax
            dots.clear();
            dots.shape.clear();

            // matmul with v
            tmp2.shape = q[h]->shape;   // {1,128,64}
            for (int pi = 0; pi < attn.shape[1]; ++pi) {
                for (int pj = 0; pj < v[h]->shape[1]; ++pj) {
                    T val2 = 0;
                    for (int pk = 0; pk < attn.shape[2]; ++pk) {
                        val2 += attn[pi * attn.shape[2] + pk] *
                            (*v[h])[pk * v[h]->shape[2] + pj];

                    }
                    tmp2.push_back(val2);
                }
            }
            attn.clear();
            attn.shape.clear();

            for(int cc_row = 0; cc_row < num; ++cc_row){      //concat matrix
                tmp3.insert(tmp3.begin() + (h * (cc_row+1) * d_k),
                // tmp3.insert(tmp3.begin() + (h * (cc_row+1) * d_k) + (cc_row * d_k),
                tmp2.begin() + (cc_row * d_k), tmp2.begin() + ((cc_row+1) * d_k)); // tmp2 line by line
            }
        }

        tmp3.shape = tmp2.shape;    // {1,128,64}
        tmp3.shape[2] = tmp3.shape[2] * heads ; // {1, 128,512}
        std::cout << tmp3 << std::endl;
        tmp2.clear();
        tmp2.shape.clear();

        // 3) final linear
        Tensor<T> tmp4{};
        outLinear->forward(tmp3, tmp4);
        tmp3.clear();
        tmp3.shape.clear();
    
        output.clear();
        output.shape.clear();
        output.insert(output.end(), tmp4.begin(), tmp4.end());
        output.shape.insert(output.shape.end(), tmp4.shape.begin(), tmp4.shape.end());
        tmp4.clear();
        tmp4.shape.clear();
    }

    ~MultiheadAttention() {
        if (qLinear != nullptr) {
            delete qLinear;
            qLinear = nullptr;
        }
        if (kLinear != nullptr) {
            delete kLinear;
            kLinear = nullptr;
        }
        if (vLinear != nullptr) {
            delete vLinear;
            vLinear = nullptr;
        }
        if (outLinear != nullptr) {
            delete outLinear;
            outLinear = nullptr;
        }
    }

    uint64_t parameterCount() override {
        uint64_t ret = 0;
        for (int i = 0; i < qLinear.size(); ++i) {
            ret += qLinear[i]->parameterCount();
            ret += kLinear[i]->parameterCount();
            ret += vLinear[i]->parameterCount();
        }
        if (outLinear != nullptr) ret += outLinear->parameterCount();
        ret += softMax.parameterCount();
        return ret;
    }

private:
    std::vector<Linear<T> *>qLinear;
    std::vector<Linear<T> *>kLinear;
    std::vector<Linear<T> *>vLinear;

    std::vector<Tensor<T> *>q;
    std::vector<Tensor<T> *>k;
    std::vector<Tensor<T> *>v;

    Linear<T> *outLinear = nullptr;
    SoftMax<T> softMax;
    int dim_model;
    int heads;
    int headDim;
    T scale;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H
