//
// Created by dianh on 2021/04/16.
//
#include "bits/stdc++.h"
#include <map>
#include <vector>
#include <string>
#include "model/attention_transformer.h"
#include "model/functions.h"

using namespace std;
typedef float data_t;

int main() {
    /*
     * Test Attention Transformer
     */
    try {
        std::cout << "Test Attention Transformer" <<std::endl;
        std::cout.flush();

        std::map<int, std::string> src_vocab;
        std::map<int, std::string> tgt_vocab;

        /* find words from tgt_vocab */
        src_vocab = text_attention::vocab_parsing("../vocab_de.txt");
        tgt_vocab = text_attention::vocab_parsing("../vocab_en.txt");

        get_param_shape("../state_dict_size.txt", text_attention::param_map);
        get_param_value("../state_dict.txt", text_attention::param_map);

        auto *attn = new text_attention::AttentionTransformer<data_t>(
                src_vocab.size(), tgt_vocab.size());

        /*Insert Sentence */
        text_attention::Tensor<data_t> input;
        text_attention::Tensor<data_t> output;

        get_param_shape( "../src_idx_size.txt", text_attention::input_idx);
        get_param_value( "../src_idx.txt", text_attention::input_idx);

        input.clear();
        input.insert(input.end(), text_attention::input_idx["src_idx"].pvals.begin(), text_attention::input_idx["src_idx"].pvals.end());
        input.shape.clear();
        input.shape = text_attention::input_idx["src_idx"].pshape; //Input Sentence padded with 128ea, 2=<blank> 
        
        std::cout << "Input Tensor Size " << input << std::endl;

        attn->forward(input, output);

        delete attn;
        std::cout << "Ok" << std::endl;
    } catch (...) {
        std::cout << "Error" << std::endl;
    }
    return 0;
}

