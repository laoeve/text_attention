//
// Created by dianh on 2021/04/16.
//
#include "bits/stdc++.h"
#include <map>
#include <vector>
#include <string>
#include <cassert>
#include "model/attention_transformer.h"
#include "model/functions.h"

using namespace std;
using namespace text_attention;
typedef float data_t;

int main(int argc, char* argv[]) {
    /* Model arguments */
    int num_layers = 6, dim_embed = 512, dim_ff = 2048, num_heads = 8;
    string path_shape = "../params/shape_selfattention.param";
    string path_value = "../params/value_selfattention.param";
    const string path_shape_input = "../sentence/shape.input";
    const string path_value_input = "../sentence/value.input";
    const string path_voca_src = "../dictionary/voca_de.dict";
    const string path_voca_tgt = "../dictinoary/voca_en.dict";

    /* Parse argument */
    for (int i=0; i<argc; ) 
    {
        std::string arg_str(argv[i]);
        if (argc==2 && (arg_str=="--help" || arg_str=="-h"))
        {
            std::cerr << "Required options are shown as below:\n"
                << "\t-l, --layers: number of layers in encoder&decoder\n"
                << "\t-e, --embedding: dimension of embedding vector\n"
                << "\t-f, --feedforward: dimension of feed-forward layer\n" 
                << "\t-m, --multiheads: number of multi-head attentions\n"
                << "\t--shape-path: path of weight parameter shapes\n"
                << "\t--value-path: path of weight parameter values\n"
                << std::endl;
            assert(0);
            exit(1);
        }

        if (i==0) 
        {
            i+=1;
            continue;
        }

        if (i+1==argc || (argv[i][0]=='-' && argv[i+1][0]=='-'))
        {
            std::cerr << "Error: no option value is available! " << std::endl;
            assert(0);
            exit(1);
        }

        if (arg_str=="-l" || arg_str=="--layers")
            num_layers = atoi(argv[i+1]);
        else if (arg_str=="-e" || arg_str=="--embedding")
            dim_embed = atoi(argv[i+1]);
        else if (arg_str=="-f" || arg_str=="--feedforward")
            dim_ff = atoi(argv[i+1]);
        else if (arg_str=="-m" || arg_str=="--multiheads")
            num_heads = atoi(argv[i+1]);
        else if (arg_str=="--shape-path") 
            path_shape = argv[i+1];
        else if (arg_str=="--value-path")
            path_value = argv[i+1];
        else
        {
            std::cerr << "Error: invalid option provided!" << std::endl;
            assert(0);
            exit(1);
        }

        i+=2;
    }

    /* Load parameters and input */
    get_param_shape(path_shape, param_map);
    get_param_value(path_value, param_map);
    get_param_shape(path_shape_input, input_idx);
    get_param_value(path_value_input, input_idx);

    /* Load vocabulary */
    map<int, string> voca_src = vocab_parsing(path_voca_src);
    map<int, string> voca_tgt = vocab_parsing(path_voca_tgt);

    /* Initialize model and input */
//    AttentionTransformer<data_t>* model = 
//        new AttentionTransformer<data_t>(num_layers, dim_embed, dim_ff, 
//                num_heads, voca_src.size( ), voca_tgt.size( ), max_len);

    std::cout << "Model initiailization complete" << std::endl;

    Tensor<data_t> input(input_idx["src_idx"].pvals, input_idx["src_idx"].pshape);
    Tensor<data_t> output;

    std::cout << "Input tensor dimension: " << input << std::endl;

    /* Run model */
    assert(0);
//    model->forward(input, output);

//    delete model;
    return 0;
}

