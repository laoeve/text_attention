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
    : Layer<T>(master), dim_model(dim_model), scale(1./sqrt(dim_model/num_heads)),
      num_heads(num_heads), headDim(dim_model/num_heads)
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
        for (int h=0; h<num_heads; h++)
        {
            /* Weight division */
            Tensor<T>* tmp_Wq = new Tensor<T>(vector<int>{dim_model,headDim});
            Tensor<T>* tmp_Wk = new Tensor<T>(vector<int>{dim_model,headDim});
            Tensor<T>* tmp_Wv = new Tensor<T>(vector<int>{dim_model,headDim});

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
            Tensor<T>* tmp_Bq = new Tensor<T>(vector<int>{headDim});
            Tensor<T>* tmp_Bk = new Tensor<T>(vector<int>{headDim});
            Tensor<T>* tmp_Bv = new Tensor<T>(vector<int>{headDim});

            sanity_cntr =0;
            for(int d_k = 0; d_k < headDim; ++d_k){
                sanity_cntr++;
                (*tmp_Bq)[d_k] = in_Bq[h*headDim+d_k];
                (*tmp_Bk)[d_k] = in_Bq[h*headDim+d_k];
                (*tmp_Bv)[d_k] = in_Bq[h*headDim+d_k];
            }
            assert(sanity_cntr==headDim);

            b_q.push_back(tmp_Bq);
            b_k.push_back(tmp_Bk);
            b_v.push_back(tmp_Bv);
            
            sanity_cntr = 0;
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

    void forward(const Tensor<T> &input, Tensor<T> &output, 
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
            for (int h=0; h<num_heads; h++)
            {
                /* Calculate matrices: Q, K, V */
                Tensor<T> mat_Q{};
                Tensor<T> mat_K{};
                Tensor<T> mat_V{};
                get_QKV(mat_Q, mat_K, mat_V, input, memory, h);

                /* Attention score (S=Q * K_t * scale) */
                Tensor<T> att_score{};
                get_attention_score(att_score, mat_Q, mat_K);

                /* Attention distribution (D=softmax[S]) */
                Tensor<T> att_dist{};
                softMax.forward(att_score, att_dist);

                /* Attention value matrix (A=D * V) */
                Tensor<T> att_val{ };
                get_attention_value(att_val, att_dist, mat_V);
                
                /* Concatenate multiple heads */
                concat_attention(mh2linear, att_val, n, h);
            }
        }

        /* Output linear */
        linear_out->forward(mh2linear, output);
    }

    ~MultiheadAttention() 
    {
        delete linear_Q;
        delete linear_K;
        delete linear_V;
        delete linear_out;
    }

    uint64_t parameterCount() override {
        uint64_t ret = 0;
        for (int i = 0; i < linear_Q.size(); ++i) {
            ret += linear_Q[i]->parameterCount();
            ret += linear_K[i]->parameterCount();
            ret += linear_V[i]->parameterCount();
        }
        if (linear_out != nullptr) ret += linear_out->parameterCount();
        ret += softMax.parameterCount();
        return ret;
    }

private:
    void get_QKV(Tensor<T>& mat_Q, Tensor<T>& mat_K, Tensor<T>& mat_V,
            const Tensor<T>& input, const Tensor<T>& memory, const int head_idx)
    {
        /* Determine output shapes */
        vector<int> out_shape{input.shape[1], headDim};
        mat_Q.reshape(out_shape);
        mat_K.reshape(out_shape);
        mat_V.reshape(out_shape);

        /* Calculate results of Q K V */
        linear_Q[head_idx]->forward(input, mat_Q);
        if (memory.is_void( ))
        {
            linear_K[head_idx]->forward(input, mat_K);
            linear_V[head_idx]->forward(input, mat_V);
        }
        else // decoder use this
        {
            linear_K[head_idx]->forward(memory, mat_K);
            linear_V[head_idx]->forward(memory, mat_V);
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
