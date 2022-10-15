//
// Created by dianh on 2021/04/16.
//
// Modified by hjpark
// based from window_attention.h

#ifndef ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H
#define ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H

#include <cmath>
#include "layer.h"
#include "tensor.h"
#include "functions.h"
#include "linear.h"
#include "softmax.h"

namespace text_attention {
    template<typename T>
    class MultiheadAttention : virtual public Layer<T> {
    public:
        MultiheadAttention(int dim_model, int heads, int max_len, std::string str_key_layer)
                : dim_model(dim_model), scale(1 / sqrt(dim_model/heads)), heads(heads), headDim(dim_model/heads) {
            std::cout << "Init.MultiheadAttention" <<std::endl<< str_key_layer <<std::endl;
            int para_num;

            for(int h = 0; h < heads; h += 1){
                para_num = 0;
                auto tmp_Wq = new Tensor<T>{};
                auto tmp_Wk = new Tensor<T>{};
                auto tmp_Wv = new Tensor<T>{};
                auto tmp_Bq = new Tensor<T>{};
                auto tmp_Bk = new Tensor<T>{};
                auto tmp_Bv = new Tensor<T>{};

                Tensor<T> para_Wq,para_Wk,para_Wv;
                para_Wq.insert(para_Wq.end(), param_map[str_key_layer+"0.weight"].pvals.begin(), param_map[str_key_layer+"0.weight"].pvals.end());
                para_Wq.shape.insert(para_Wq.shape.end(), param_map[str_key_layer+"0.weight"].pshape.begin(), param_map[str_key_layer+"0.weight"].pshape.end());
                t_tran(para_Wq);
                para_Wk.insert(para_Wk.end(), param_map[str_key_layer+"1.weight"].pvals.begin(), param_map[str_key_layer+"1.weight"].pvals.end());
                para_Wk.shape.insert(para_Wk.shape.end(), param_map[str_key_layer+"1.weight"].pshape.begin(), param_map[str_key_layer+"1.weight"].pshape.end());
                t_tran(para_Wk);
                para_Wv.insert(para_Wv.end(), param_map[str_key_layer+"2.weight"].pvals.begin(), param_map[str_key_layer+"2.weight"].pvals.end());
                para_Wv.shape.insert(para_Wv.shape.end(), param_map[str_key_layer+"2.weight"].pshape.begin(), param_map[str_key_layer+"2.weight"].pshape.end());
                t_tran(para_Wv);

                for(int line = 0; line < dim_model; ++line){
                    for(int d_k = 0; d_k < headDim; ++d_k){
                        para_num++;
                        tmp_Wq->push_back(para_Wq[line * dim_model + h * headDim + d_k]);
                        tmp_Wk->push_back(para_Wk[line * dim_model + h * headDim + d_k]);
                        tmp_Wv->push_back(para_Wv[line * dim_model + h * headDim + d_k]);
                    }
                }
                assert(para_num == dim_model * headDim);

/*                 std::cout << "tmp Wq : "; for(auto i : tmp_Wq->shape){std::cout << i <<" ";}; std::cout << std::endl;
                std::cout << "tmp Wk : "; for(auto i : tmp_Wk->shape){std::cout << i <<" ";}; std::cout<< std::endl;
                std::cout << "tmp Wv : "; for(auto i : tmp_Wv->shape){std::cout << i <<" ";}; std::cout<< std::endl; */
                tmp_Wq->shape = {dim_model, headDim};
                tmp_Wk->shape = {dim_model, headDim};
                tmp_Wv->shape = {dim_model, headDim};
                std::cout << "Verify pasing heads : " << *tmp_Wq << std::endl;
                w_q.push_back(tmp_Wq);
                w_k.push_back(tmp_Wk);
                w_v.push_back(tmp_Wv);

                para_num =0;
                for(int d_k = 0; d_k < headDim; ++d_k){
                    para_num++;
                    tmp_Bq->push_back(param_map[str_key_layer+"0.bias"].pvals[h * headDim + d_k]);
                    tmp_Bk->push_back(param_map[str_key_layer+"1.bias"].pvals[h * headDim + d_k]);
                    tmp_Bv->push_back(param_map[str_key_layer+"2.bias"].pvals[h * headDim + d_k]);
                }
                assert(para_num == headDim);

/*                 std::cout << "tmp Bq : "; for(auto i : tmp_Bq->shape){std::cout << i <<" ";}; std::cout<< std::endl;
                std::cout << "tmp Bk : "; for(auto i : tmp_Bk->shape){std::cout << i <<" ";}; std::cout<< std::endl;
                std::cout << "tmp Bv : "; for(auto i : tmp_Bv->shape){std::cout << i <<" ";}; std::cout<< std::endl; */

                tmp_Bq->shape = {headDim};
                tmp_Bk->shape = {headDim};
                tmp_Bv->shape = {headDim};
                b_q.push_back(tmp_Bq);
                b_k.push_back(tmp_Bk);
                b_v.push_back(tmp_Bv);

                para_Wq.clear();
                para_Wk.clear();
                para_Wv.clear();
            }

/*             std::cout << "Linear Element Size :" << std::endl;
            for(auto it : (*b_q[0]).shape){std::cout << " "<< it;};
            std::cout << std::endl;

            std::cout << "Linear Element :" << std::endl;
            int num_shape =1;
            for(int it = 0 ; it < (*b_q[0]).shape.size() ; ++ it){num_shape *= (*b_q[0]).shape[it];};
            for(int its = 0 ; its < num_shape+1 ; ++its  ){std::cout << " " <<(*b_q[0])[its];};
            std::cout << std::endl; */

            for(int h = 0; h < heads; h +=1){
                auto lin_q = new Linear<T>(dim_model, headDim, *w_q[h], *b_q[h]);
                auto lin_k = new Linear<T>(dim_model, headDim, *w_k[h], *b_k[h]);
                auto lin_v = new Linear<T>(dim_model, headDim, *w_v[h], *b_v[h]);
                qLinear.push_back(lin_q);   //{1,128,64}
                kLinear.push_back(lin_k);
                vLinear.push_back(lin_v);
            }

            auto w_out = new Tensor<T>{};
            w_out->insert(w_out->end(), param_map[str_key_layer+"3.weight"].pvals.begin(), param_map[str_key_layer+"3.weight"].pvals.end());
            w_out->shape.insert(w_out->shape.end(), param_map[str_key_layer+"3.weight"].pshape.begin(), param_map[str_key_layer+"3.weight"].pshape.end());
            t_tran(*w_out);
            auto b_out = new Tensor<T>{};
            b_out->insert(b_out->end(), param_map[str_key_layer+"3.bias"].pvals.begin(), param_map[str_key_layer+"3.bias"].pvals.end());
            b_out->shape.insert(b_out->shape.end(), param_map[str_key_layer+"3.bias"].pshape.begin(), param_map[str_key_layer+"3.bias"].pshape.end());

            outLinear = new Linear<T>(dim_model, dim_model, *w_out, *b_out);
            //std::random_device rd{};
            //std::mt19937 gen{rd()};
            //std::normal_distribution<> d{0, 1};
        }


        void forward(const Tensor<T> &input, Tensor<T> &output, Tensor<T> &mask, Tensor<T> &memory) {
            std::cout << "MultiheadAttention.Forward" << std::endl;
            //int dim_k = this->dim_model / this->heads;
            Tensor<T> tmp1{};

            // lin(x) for lin, x in zip(self.linears, (query, key, value))
            tmp1.insert(tmp1.end(), input.begin(), input.end());
            tmp1.shape.insert(tmp1.shape.end(), input.shape.begin(), input.shape.end());    //{1,128,512}
            std::cout << tmp1 << std::endl;

            if(&memory != nullptr){
                for(auto item : (this->qLinear)){
                    auto q_tmp = new Tensor<T>{};
                    item->forward(memory, *q_tmp);
                    q.push_back(q_tmp);
                }
                for(auto item : (this->kLinear)){
                    auto k_tmp = new Tensor<T>{};
                    item->forward(memory, *k_tmp);
                    k.push_back(k_tmp);
                }
                for(auto item : (this->vLinear)){
                    auto v_tmp = new Tensor<T>{};
                    item->forward(tmp1, *v_tmp);
                    v.push_back(v_tmp);
                }
            } else {
                for(auto item : (this->qLinear)){
                    auto q_tmp = new Tensor<T>{};
                    item->forward(tmp1, *q_tmp);
                    q.push_back(q_tmp);
                }
                for(auto item : (this->kLinear)){
                    auto k_tmp = new Tensor<T>{};
                    item->forward(tmp1, *k_tmp);
                    k.push_back(k_tmp);
                }
                for(auto item : (this->vLinear)){
                    auto v_tmp = new Tensor<T>{};
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

        long long parameterCount() {
            long long ret = 0;
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

        std::vector<Tensor<T> *>w_q;
        std::vector<Tensor<T> *>w_k;
        std::vector<Tensor<T> *>w_v;
        
        std::vector<Tensor<T> *>b_q;
        std::vector<Tensor<T> *>b_k;
        std::vector<Tensor<T> *>b_v;

        Linear<T> *outLinear = nullptr;
        SoftMax<T> softMax;
        int dim_model;
        int heads;
        int headDim;
        T scale;
    };
}
#endif //ATTENTION_TRANSFORMER_CPP_MULTIHEADATTENTION_H
