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
        string wout_str = prefix_str+"."+sa_out_str+"."+weight_str; // h.0 . attn.c_proj . weight
        string bout_str = prefix_str+"."+sa_out_str+"."+bias_str;

        /* 'Tensorize' */

        Tensor<T> in_Wq { };
        Tensor<T> in_Wk { };
        Tensor<T> in_Wv { };
        Tensor<T> in_Bq { };
        Tensor<T> in_Bk { };
        Tensor<T> in_Bv { };

        if (sa_query_str != "attn.c_attn")
        {
            Tensor<T> in_Wq(param_map[wq_str].pvals, param_map[wq_str].pshape);
            Tensor<T> in_Wk(param_map[wk_str].pvals, param_map[wk_str].pshape);
            Tensor<T> in_Wv(param_map[wv_str].pvals, param_map[wv_str].pshape);
            Tensor<T> in_Bq(param_map[bq_str].pvals, param_map[bq_str].pshape);
            Tensor<T> in_Bk(param_map[bk_str].pvals, param_map[bk_str].pshape);
            Tensor<T> in_Bv(param_map[bv_str].pvals, param_map[bv_str].pshape);
        }
        else
        {
            /* case for gpt2 */
            string weight_str = prefix_str+"."+sa_query_str+"."+weight_str;
            string bias_str = prefix_str+"."+sa_query_str+"."+bias_str;

            Tensor<T> weight_merge(param_map[weight_str].pvals, param_map[weight_str].pshape);
            Tensor<T> bias_merge(param_map[bias_str].pvals, param_map[bias_str].pshape);

            split_weight_QKV(in_Wq,in_Wk,in_Wv, weight_merge , dim_model);
            split_bias_QKV(in_Bq,in_Bk,in_Bv, bias_merge , dim_model);
        }
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
            Tensor<T>* tmp_Wq = new Tensor<T>(vector<int>{512,64});
            Tensor<T>* tmp_Wk = new Tensor<T>(vector<int>{512,64});
            Tensor<T>* tmp_Wv = new Tensor<T>(vector<int>{512,64});

            for (int line=0; line<dim_model; line++)
            {
                for (int d_k=0; d_k<headDim; d_k++)
                {
                    sanity_cntr++;
                    (*tmp_Wq)[line*dim_model+d_k] = in_Wq[line*dim_model+h*headDim+d_k];
                    (*tmp_Wk)[line*dim_model+d_k] = in_Wk[line*dim_model+h*headDim+d_k];
                    (*tmp_Wv)[line*dim_model+d_k] = in_Wv[line*dim_model+h*headDim+d_k];
                }
            }
            assert(sanity_cntr==dim_model * headDim);

            w_q.push_back(tmp_Wq);
            w_k.push_back(tmp_Wk);
            w_v.push_back(tmp_Wv);

            /* Bias division */
            Tensor<T>* tmp_Bq = new Tensor<T>(vector<int>{64});
            Tensor<T>* tmp_Bk = new Tensor<T>(vector<int>{64});
            Tensor<T>* tmp_Bv = new Tensor<T>(vector<int>{64});

            for (int i=0; i<headDim; i++)
            {
                (*tmp_Bq)[i] = in_Bq[h*headDim+i];
                (*tmp_Bk)[i] = in_Bk[h*headDim+i];
                (*tmp_Bv)[i] = in_Bv[h*headDim+i];
            }
            assert(sanity_cntr==headDim);

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

            qLinear.push_back(lin_q);
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
        Tensor<T> tmp1{};
        tmp1 = input;
        
        // 1) Linear FC Layer Projection
        // lin(x) for lin, x in zip(self.linears, (query, key, value))

        if(memory.is_void( )==false)
        {
            for(auto item : (this->qLinear))
            {
                Tensor<T>* q_tmp = new Tensor<T>{};
                item->forward(tmp1, *q_tmp);
                q_fcm.push_back(q_tmp);
            }
            for(auto item : (this->kLinear))
            {
                Tensor<T>* k_tmp = new Tensor<T>{};
                item->forward(memory, *k_tmp);
                k_fcm.push_back(k_tmp);
            }
            for(auto item : (this->vLinear))
            {
                Tensor<T>* v_tmp = new Tensor<T>{};
                item->forward(memory, *v_tmp);
                v_fcm.push_back(v_tmp);
            }
        }
        else
        {
            for(auto item : (this->qLinear))
            {
                Tensor<T>* q_tmp = new Tensor<T>{};
                item->forward(tmp1, *q_tmp);
                q_fcm.push_back(q_tmp);
            }
            for(auto item : (this->kLinear))
            {
                Tensor<T>* k_tmp = new Tensor<T>{};
                item->forward(tmp1, *k_tmp);
                k_fcm.push_back(k_tmp);
            }
            for(auto item : (this->vLinear))
            {
                Tensor<T>* v_tmp = new Tensor<T>{};
                item->forward(tmp1, *v_tmp);
                v_fcm.push_back(v_tmp);
            }
        }

private:

    void split_weight_QKV(Tensor<T>& wmx_Q, Tensor<T>& wmx_K, Tensor<T>& wmx_V,
        const Tensor<T>& input, const int dim_model)
    {
        wmx_Q.reshape(std::vector<int>{dim_model,dim_model});
        wmx_K.reshape(std::vector<int>{dim_model,dim_model});
        wmx_V.reshape(std::vector<int>{dim_model,dim_model});

        for (int i=0; i<dim_model*3; i++)
        {
            for (int j=0; j<dim_model; j++)
            {
                wmx_Q[i*dim_model+j] = input[i*dim_model*3+j];
                wmx_K[i*dim_model+j] = input[i*dim_model*3+dim_model*1+j];
                wmx_V[i*dim_model+j] = input[i*dim_model*3+dim_model*2+j];
            }
        }
    }

    void split_bias_QKV(Tensor<T>& bmx_Q, Tensor<T>& bmx_K, Tensor<T>& bmx_V,
        const Tensor<T>& input, const int dim_model)
    {
        bmx_Q.reshape(std::vector<int>{dim_model});
        bmx_K.reshape(std::vector<int>{dim_model});
        bmx_V.reshape(std::vector<int>{dim_model});
        
        for (int i=0; i<dim_model; i++)
        {
            bmx_Q[i] = input[i];
            bmx_K[i] = input[dim_model*1+i];
            bmx_V[i] = input[dim_model*2+i];
        }
    }

    void split_batch_layer(Tensor<T>& single_input, 
            const Tensor<T>& input, const int input_idx)
    {
        if (input.is_void( ))
            return;

        Tensor<T> dots(vector<int> {1,word_num,word_num});
        Tensor<T> tmp2{};
        Tensor<T> tmp3(vector<int> {1,word_num,word_num});
        Tensor<T> tmp4{};
        Tensor<T> attn{};

        // 2) scale dot attention
        // matmul(q, k^t)
        for (int h = 0; h < heads; ++h)
        {
            for (int pi = 0; pi < word_num; ++pi)
            {
                for (int pj = 0; pj < word_num; ++pj)
                {
                    T val = 0;
                    for (int pk = 0; pk < d_k; ++pk)
                    {
                        val += (*q_fcm[h])[pi * d_k + pk]
                         * (*k_fcm[h])[pj * d_k + pk]
                         * scale;
                    }
                    dots[pi*word_num + pj] = val;
                }
            }

            if (mask.is_void( )==false)
            {
                if(mask.shape[mask.shape.size()-1] == mask.shape[mask.shape.size()-2])
                {   //condition : src_mask {1 * num}
                    for (int p = 0; p < mask.shape[1]; ++p)
                    {   
                        for (int j = 0; j < mask.shape[2]; ++j)
                        {
                            if(mask[j] == false)
                            {
                                dots[mask.shape[2] * p + j] = -1e9;
                            }
                        }
                    }
                } else { //condition : tgt_mask {num * num}
                    for (int p = 0; p < mask.shape[1]; ++p)
                    {
                        for (int j = 0; j < mask.shape[2]; ++j)
                        {
                            if(mask[mask.shape[2] * p + j] == false)
                            {
                                dots[mask.shape[2] * p + j] = -1e9;
                            }
                        }
                    }
                }
            }

            softMax.forward(dots, attn);

            /* matmul with v */
            tmp2.reshape(q_fcm[h]->shape);   // {1,num,headDim}
            for (int pi = 0; pi < attn.shape[1]; ++pi)
            {
                for (int pj = 0; pj < v_fcm[h]->shape[1]; ++pj)
                {
                    T val2 = 0;
                    for (int pk = 0; pk < attn.shape[2]; ++pk)
                    {
                        val2 += attn[pi * attn.shape[2] + pk] *
                            (*v_fcm[h])[pk * v_fcm[h]->shape[2] + pj];
                    }
                    tmp2[v_fcm[h]->shape[1]]=val2;
                }
            }

            std::vector<int> out_shape = tmp2.shape;
            out_shape[tmp2.shape.size() -1] = tmp2.shape[tmp2.shape.size() -1] * heads;
            tmp3.reshape(out_shape);    // {1, word_num, dim_model}

            /* Concat Head matrix */
            for(int cc_row = 0; cc_row < word_num; ++cc_row)
            {    // num : len_word
                for(int cc_col = 0; cc_col < d_k ; ++ cc_col)
                {
                    tmp3[dim_model*cc_row + h*d_k + cc_col] = tmp2[d_k*cc_row + cc_col];
                }
            }
        }

        // 3) final linear
        outLinear->forward(tmp3, tmp4);
        output = tmp4;
    }

    ~MultiheadAttention() 
    {
        delete qLinear;
        delete kLinear;
        delete vLinear;
        delete outLinear;
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

    std::vector<Tensor<T> *>q_fcm; //fully connected matrix
    std::vector<Tensor<T> *>k_fcm;
    std::vector<Tensor<T> *>v_fcm;

    Linear<T> *outLinear = nullptr;
    SoftMax<T> softMax;
    int dim_model;
    int heads;
    int headDim;
    bool div_weight;
    T scale;
};
}
#endif //ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H
