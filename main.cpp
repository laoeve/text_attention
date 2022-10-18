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

void swap_tensor(Tensor<data_t>& swp_a, Tensor<data_t>& swp_b)
{
    Tensor<data_t> swp_tmp = swp_b;
    swp_b = swp_a;
    swp_a = swp_tmp;
}

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

    vector<int> opa_shape{3, 2, 3};
    Tensor<data_t> opa(opa_shape);
    opa[0] = 0; opa[1] = 1; opa[2] = 2; 
    opa[3] = 3; opa[4] = 4; opa[5] = 5;

    opa[6] = 5; opa[7] = 4; opa[8] = 3; 
    opa[9] = 2; opa[10] = 1; opa[11] = 0;

    opa[12] = 9; opa[13] = 8; opa[14] = 7; 
    opa[15] = 6; opa[16] = 5; opa[17] = 4;

    vector<int> opw_shape{3, 4};
    Tensor<data_t>* opw = new Tensor<data_t>(opw_shape);
    (*opw)[0] = 1; (*opw)[1] = 2; (*opw)[2] = 3; (*opw)[3] = 4;
    (*opw)[4] = 5; (*opw)[5] = 6; (*opw)[6] = 7; (*opw)[7] = 8;
    (*opw)[8] = 9; (*opw)[9] = 10; (*opw)[10] = 11; (*opw)[11] = 12;

    vector<int> opb_shape{4};
    Tensor<data_t>* opb = new Tensor<data_t>(opb_shape);
    (*opb)[0] = 7;
    (*opb)[1] = 7;
    (*opb)[2] = 7;
    (*opb)[3] = 7;

    Linear<data_t>* lin = new Linear<data_t>("testlinear", 3, 4, *opw, *opb);

    Tensor<data_t> out;
    lin->forward(opa, out);
    opa.print_all( );
    out.print_all( );

    vector<data_t> gamma{5, 9, 4, 7};
    vector<data_t> beta{3, 1, 2, 9};

    Tensor<data_t> out_ln;
    LayerNorm<data_t>* ln = new LayerNorm<data_t>("testln", 4, gamma, beta);
    ln->forward(out, out_ln);
    out_ln.print_all( );

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

