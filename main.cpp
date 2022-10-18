//
// Created by dianh on 2021/04/16.
//
#include "bits/stdc++.h"
#include <map>
#include <vector>
#include <string>
#include <cassert>
#include "model/top_model.h"
#include "model/attention_transformer.h"
#include "model/functions.h"

using namespace std;
using namespace text_attention;
typedef float data_t;

int main(int argc, char* argv[]) {
    /* Model arguments */
    string path_shape_input = "../sentence/shape.input";
    string path_value_input = "../sentence/value.input";
    string path_voca_src = "../dictionary/voca_de.dict";
    string path_voca_tgt = "../dictionary/voca_en.dict";
    string model_arg = "transformer";

    /* Parse argument */
    for (int i=0; i<argc; ) 
    {
        std::string arg_str(argv[i]);
        if (argc==2 && (arg_str=="--help" || arg_str=="-h"))
        {
            std::cerr << "Required options are shown as below:\n"
                << "\t-m, --model: name of model "
                << "[transformer, bert-base, bert-large, gpt2]\n"
                << "\t--input-shape-path: path of input shape\n"
                << "\t--input-value-path: path of input shape\n"
                << std::endl;
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

        if (arg_str=="-m" || arg_str=="--model")
            model_arg = argv[i+1];
        else if (arg_str=="--input-shape-path") 
            path_shape_input = argv[i+1];
        else if (arg_str=="--input-value-path")
            path_value_input = argv[i+1];
        else
        {
            std::cerr << "Error: invalid option provided!" << std::endl;
            assert(0);
            exit(1);
        }

        i+=2;
    }

    /* Load vocabulary */
    map<int, string> voca_src = vocab_parsing(path_voca_src);
    map<int, string> voca_tgt = vocab_parsing(path_voca_tgt);

    std::cout << "source dictionary size: " << voca_src.size( ) << std::endl;
    std::cout << "target dictinoary size: " << voca_tgt.size( ) << std::endl;

    Tensor<data_t> test(vector<int>{1, 4});
    test[0] = 3; test[1] = 5; test[2] = 7; test[3] = 33;
    std::cout << "CHECK " << test << std::endl;
    test.reshape(vector<int>{1, 5});
    std::cout << "CHECK " << test << std::endl;
    test[4] = 99;
    std::cout << "CHECK " << test << std::endl;



    assert(0);

    /* Init models */
    TopModel<data_t>* model = nullptr;
    if (model_arg=="transformer") 
    { 
        string path_shape = "../params/shape_transformer.param";
        string path_value = "../params/value_transformer.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
        model = new AttentionTransformer<data_t>(voca_src.size( ), voca_tgt.size( ));
    }
    else if (model_arg=="bert-base")
    {
        //TODO
        assert(0);
        string path_shape = "../params/shape_bert-base.param";
        string path_value = "../params/value_bert-base.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
    }
    else if (model_arg=="bert-large")
    {
        //TODO
        assert(0);
        string path_shape = "../params/shape_bert-large.param";
        string path_value = "../params/value_bert-large.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
    }
    else if (model_arg=="gpt2")
    {
        //TODO
        assert(0);
        string path_shape = "../params/shape_gpt2.param";
        string path_value = "../params/value_gpt2.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
    }
    else
    {
        std::cerr << "Error: unavailable models" << std::endl;
        assert(0);
        exit(1);
    }
    std::cout << "Model initiailization complete" << std::endl;

    /* Load input */
    get_param_shape(path_shape_input, input_idx);
    get_param_value(path_value_input, input_idx);

    Tensor<data_t> input(input_idx["src_idx"].pvals, input_idx["src_idx"].pshape);
    Tensor<data_t> output;
    std::cout << "Input tensor dimension: " << input << std::endl;

    /* Run model */
//    model->forward(input, output);

//    delete model;
    return 0;
}

