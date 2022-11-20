/*
 * Copyright (c) 2022 Computer Architecture and Paralllel Processing Lab, 
 * Seoul National University, Republic of Korea. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     1. Redistribution of source code must retain the above copyright 
 *        notice, this list of conditions and the follwoing disclaimer.
 *     2. Redistributions in binary form must reproduce the above copyright 
 *        notice, this list conditions and the following disclaimer in the 
 *        documentation and/or other materials provided with the distirubtion.
 *     3. Neither the name of the copyright holders nor the name of its 
 *        contributors may be used to endorse or promote products derived from 
 *        this software without specific prior written permission. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 
 * Hyokeun Lee (hklee@capp.snu.ac.kr)
 * Hyunjun Park (laoeve@capp.snu.ac.kr)
 *
 */

#ifndef ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H
#define ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H

#include "bits/stdc++.h"
#include "top_model.h"
#include "layer.h"
#include "tensor.h"
#include "functions.h"
#include "linear.h"
#include "softmax.h"

#define MULTIHEAD_DBG

using namespace std;

namespace text_attention {
template<typename T>
class MultiheadAttention : virtual public Layer<T> 
{
public:
    MultiheadAttention(TopModel<T>* master,
            int dim_model, int num_heads, const string prefix_str,
            const string weight_str, const string bias_str, 
            const string sa_query_str, const string sa_key_str,
            const string sa_value_str, const string sa_out_str)
    : Layer<T>(master), dim_model(dim_model), scale(1./sqrt(dim_model/num_heads)),
      num_heads(num_heads), headDim(dim_model/num_heads)
    {
        std::cout << ">>>> Init multihead sublayer - " << std::endl;

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
        Tensor<T>* in_Wq = new Tensor<T> { };
        Tensor<T>* in_Wk = new Tensor<T> { };
        Tensor<T>* in_Wv = new Tensor<T> { };
        Tensor<T>* in_Bq = new Tensor<T> { };
        Tensor<T>* in_Bk = new Tensor<T> { };
        Tensor<T>* in_Bv = new Tensor<T> { };

        if (sa_query_str != "attn.c_attn")
        {
            in_Wq = new Tensor<T>(param_map[wq_str].pvals, param_map[wq_str].pshape);
            in_Wk = new Tensor<T>(param_map[wk_str].pvals, param_map[wk_str].pshape);
            in_Wv = new Tensor<T>(param_map[wv_str].pvals, param_map[wv_str].pshape);
            if (bias_str.empty( ) == false)
            {
                in_Bq = new Tensor<T>(param_map[bq_str].pvals, param_map[bq_str].pshape);
                in_Bk = new Tensor<T>(param_map[bk_str].pvals, param_map[bk_str].pshape);
                in_Bv = new Tensor<T>(param_map[bv_str].pvals, param_map[bv_str].pshape);
            }
        }
        else
        {
            /* case for gpt2 */
            string weight_gpt2_str = prefix_str+"."+sa_query_str+"."+weight_str;
            string bias_gpt2_str = prefix_str+"."+sa_query_str+"."+bias_str;

            Tensor<T> weight_merge(param_map[weight_gpt2_str].pvals, param_map[weight_gpt2_str].pshape);
            Tensor<T> bias_merge(param_map[bias_gpt2_str].pvals, param_map[bias_gpt2_str].pshape);

            split_weight_QKV(*in_Wq,*in_Wk,*in_Wv, weight_merge , dim_model);
            split_bias_QKV(*in_Bq,*in_Bk,*in_Bv, bias_merge , dim_model);
        }

        Tensor<T>* w_out = 
            new Tensor<T>(param_map[wout_str].pvals, param_map[wout_str].pshape);
        
        Tensor<T>* b_out = nullptr;
        if (bias_str.empty( ))
        {
            b_out = new Tensor<T> { };
        }
        else
        {
            b_out = new Tensor<T>(param_map[bout_str].pvals, param_map[bout_str].pshape);
        }

        /* Divide parametes into multiple heads */
        std::vector<Tensor<T> *>w_q;
        std::vector<Tensor<T> *>w_k;
        std::vector<Tensor<T> *>w_v;
        std::vector<Tensor<T> *>b_q;
        std::vector<Tensor<T> *>b_k;
        std::vector<Tensor<T> *>b_v;
        for (int h=0; h<num_heads; h++)
        {
            /* Weight division */
            Tensor<T>* tmp_Wq = new Tensor<T>(vector<int>{dim_model,headDim});
            Tensor<T>* tmp_Wk = new Tensor<T>(vector<int>{dim_model,headDim});
            Tensor<T>* tmp_Wv = new Tensor<T>(vector<int>{dim_model,headDim});

            for (int i=0; i<dim_model; i++)
            {
                for (int j=0; j<headDim; j++)
                {
                    (*tmp_Wq)[i*headDim+j] = (*in_Wq)[i*dim_model+h*headDim+j];
                    (*tmp_Wk)[i*headDim+j] = (*in_Wk)[i*dim_model+h*headDim+j];
                    (*tmp_Wv)[i*headDim+j] = (*in_Wv)[i*dim_model+h*headDim+j];
                }
            }

            w_q.push_back(tmp_Wq);
            w_k.push_back(tmp_Wk);
            w_v.push_back(tmp_Wv);

            /* Bias division */
            Tensor<T>* tmp_Bq = nullptr;
            Tensor<T>* tmp_Bk = nullptr;
            Tensor<T>* tmp_Bv = nullptr;

            if (bias_str.empty( ))
            {
                tmp_Bq = new Tensor<T> { };
                tmp_Bk = new Tensor<T> { };
                tmp_Bv = new Tensor<T> { };
            }
            else
            {
                tmp_Bq = new Tensor<T>(vector<int>{headDim});
                tmp_Bk = new Tensor<T>(vector<int>{headDim});
                tmp_Bv = new Tensor<T>(vector<int>{headDim});

                for (int i=0; i<headDim; i++)
                {
                    (*tmp_Bq)[i] = (*in_Bq)[h*headDim+i];
                    (*tmp_Bk)[i] = (*in_Bk)[h*headDim+i];
                    (*tmp_Bv)[i] = (*in_Bv)[h*headDim+i];
                }
            }
            b_q.push_back(tmp_Bq);
            b_k.push_back(tmp_Bk);
            b_v.push_back(tmp_Bv);
        }

        /* Init linear layers */
        for (int h=0; h<num_heads; h++)
        {
            Linear<T>* tmp_q = new Linear<T>(
                    prefix_str+"."+sa_query_str+".HD["+to_string(h)+"]", 
                    dim_model, headDim, *w_q[h], *b_q[h]);
            Linear<T>* tmp_k = new Linear<T>(
                    prefix_str+"."+sa_key_str+".HD["+to_string(h)+"]", 
                    dim_model, headDim, *w_k[h], *b_k[h]);
            Linear<T>* tmp_v = new Linear<T>(
                    prefix_str+"."+sa_value_str+".HD["+to_string(h)+"]", 
                    dim_model, headDim, *w_v[h], *b_v[h]);

            linear_Q.push_back(tmp_q);
            linear_K.push_back(tmp_k);
            linear_V.push_back(tmp_v);

            tmp_q->print_params( );
            tmp_k->print_params( );
            tmp_v->print_params( );
        }
        
        linear_out = new Linear<T>(prefix_str+"."+sa_out_str,
                dim_model, dim_model, *w_out, *b_out);
        linear_out->print_params( );
    }

    void forward(Tensor <T> &output, const Tensor<T> &input,
            const Tensor<bool> &mask, const Tensor<T> &memory) override 
    {
       if (is_operable(input)==false)
        {
            std::cerr << "Error: dimension error on MultiHead layer " << std::endl;
            assert(0);
            exit(1);
        }

        int num_input = input.shape[0];
        int num_row = input.shape[1];

        Tensor<T> mh2linear;
        vector<int> out_shape{num_input, num_row, dim_model};
        mh2linear.reshape(out_shape);

        for (int n=0; n<num_input; n++)
        {
            /* Extract a single input for 2D calculation */
            Tensor<T> single_input{ };
            Tensor<T> single_memory{ };
            split_batch_layer(single_input, input, n);
            split_batch_layer(single_memory, memory, n);

            for (int h=0; h<num_heads; h++)
            {
                /* Calculate matrices: Q, K, V */
                Tensor<T> mat_Q{};
                Tensor<T> mat_K{};
                Tensor<T> mat_V{};
                get_QKV(mat_Q, mat_K, mat_V, single_input, single_memory, h);

                /* Attention score (S=Q * K_t * scale) */
                Tensor<T> att_score{};
                get_attention_score(att_score, mat_Q, mat_K);

                /* Attention distribution (D=softmax[S]) */
                Tensor<T> att_dist{};
                get_attention_dist(att_dist, att_score, mask, n);

                /* Attention value matrix (A=D * V) */
                Tensor<T> att_val{ };
                get_attention_value(att_val, att_dist, mat_V);
                
                /* Concatenate multiple heads */
                concat_attention(mh2linear, att_val, n, h);
            }
        }

        /* Output linear */
        linear_out->forward(output, mh2linear);
    }

    ~MultiheadAttention() 
    {
        delete linear_Q;
        delete linear_K;
        delete linear_V;
        delete linear_out;
    }

    uint64_t parameterCount() override 
    {
        uint64_t ret = 0;
        for (int i = 0; i < linear_Q.size(); ++i) 
        {
            ret += linear_Q[i]->parameterCount();
            ret += linear_K[i]->parameterCount();
            ret += linear_V[i]->parameterCount();
        }
        if (linear_out != nullptr) ret += linear_out->parameterCount();
        ret += softMax.parameterCount();
        return ret;
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

        int num_row = input.shape[1];
        int num_col = input.shape[2];

        single_input.reshape(std::vector<int>{num_row, num_col});
        for (int i=0; i<num_row; i++)
        {
            for (int j=0; j<num_col; j++)
            {
                single_input[i*num_col+j] = 
                    input[input_idx*num_row*num_col+i*num_col+j];
            }
        }
    }

    void get_QKV(Tensor<T>& mat_Q, Tensor<T>& mat_K, Tensor<T>& mat_V,
            const Tensor<T>& input, const Tensor<T>& memory, const int head_idx)
    {
        /* Calculate results of Q K V */
        linear_Q[head_idx]->forward(mat_Q, input);
        if (memory.is_void( ))
        {
            linear_K[head_idx]->forward(mat_K, input);
            linear_V[head_idx]->forward(mat_V, input);
        }
        else // decoder use this
        {
            linear_K[head_idx]->forward(mat_K, memory);
            linear_V[head_idx]->forward(mat_V, memory);
        }
    }

    void get_attention_score(Tensor<T>& att_score, 
            const Tensor<T>& mat_Q, Tensor<T>& mat_K)
    {
        /* Transpose K */
        mat_K.transpose( );

        /* Matrix multiplication */
        assert(mat_Q.shape[1]==headDim);
        Layer<T>::matmul(att_score, mat_Q, mat_K, scale);
    }

    void get_attention_dist(Tensor<T>& att_dist, Tensor<T>& att_score, 
            const Tensor<bool>& mask, const int input_idx)
    {
        /* Mask out values */
        if (mask.is_void( )==false)
        {
            if (mask.get_dims( )==2)
            {
                /* Target mask */
#ifdef DEBUG
                if (att_score.shape[0]!=mask.shape[0] ||
                    att_score.shape[1]!=mask.shape[1] ||
                    mask.shape[0]!=mask.shape[1])
                {
                    std::cerr << "Error: dimension error while getting"
                        << " target masked results" << std::endl;
                    assert(0);
                    exit(1);
                }
#endif

                uint64_t len = att_score.shape[0];
                for (int i=0; i<len; i++)
                {
                    for (int j=0; j<len; j++)
                    {
                        if (mask[i*len+j])
                            continue;
                        att_score[i*len+j] = -1e9;
                    } 
                }
            }
            else
            {
                /* Source/encoder-attention mask */
#ifdef DEBUG
                if (mask.get_dims( )!=3 ||
                    att_score.shape[0]!=mask.shape[1] ||
                    att_score.shape[1]!=mask.shape[2])
                {
                    std::cerr << "Error: dimension error while getting"
                        << " encoder masked result" << std::endl;
                    assert(0);
                    exit(1);
                }
#endif
                uint64_t len_tgt = mask.shape[1];
                uint64_t len_enc = mask.shape[2];
                uint64_t offset = input_idx*len_tgt*len_enc;
                for (int i=0; i<len_tgt; i++)
                {
                    for (int j=0; j<len_enc; j++)
                    {
                        if (mask[offset+i*len_enc+j])
                            continue;

                        att_score[i*len_enc+j] = -1e9;
                    }
                }
            }
        }

        /* Calculate distribution */
        softMax.forward(att_dist, att_score);
    }

    void get_attention_value(Tensor<T>& att_val,
            const Tensor<T>& att_dist, const Tensor<T>& mat_V)
    {
        Layer<T>::matmul(att_val, att_dist, mat_V, 1.0);
    }

    void concat_attention(Tensor<T>& out, const Tensor<T>& att_val, 
            const int input_idx, const int head_idx)
    {
        assert(out.get_dims( )==3);

        uint64_t sz_stack = out.shape[1]*out.shape[2];
        uint64_t v_row = att_val.shape[0];
        uint64_t v_col = att_val.shape[1];
        uint64_t o_col = out.shape[2];
        for (uint64_t v_i=0; v_i<v_row; v_i++)
        {
            for (uint64_t v_j=0; v_j<v_col; v_j++)
            {
                uint64_t o_j = head_idx*v_col+v_j;
                out[input_idx*sz_stack+v_i*o_col+o_j] = att_val[v_i*v_col+v_j];
            }
        }
    }

    bool is_operable(const Tensor<T>& op)
    {
        /* 
         * Dimension of '3' means multiple inputs 
         * op can be seen as [#INPUT x #WORDS x DIM_EMBED]
         */
        if (op.get_dims( )!=3 || op.shape[2]!=dim_model)
            return false;
        
        return true;
    }

    std::vector<Linear<T> *>linear_Q;
    std::vector<Linear<T> *>linear_K;
    std::vector<Linear<T> *>linear_V;

    Linear<T> *linear_out = nullptr;
    SoftMax<T> softMax;
    int dim_model;
    int num_heads;
    int headDim;
    T scale;

};
}
#endif //ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H