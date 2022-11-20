//
// Created by dianh on 2021/04/16.
//
#include "bits/stdc++.h"
#include <cassert>
#include <chrono>
#include "model/top_model.h"
#include "model/functions.h"

//TODO: add header of models on the following line
#include "model/attention_transformer.h"
#include "model/bert.h"
#include "model/gpt2.h"
#include "model/t5.h"

using namespace std;
using namespace text_attention;
typedef float data_t;

int main(int argc, char* argv[]) 
{
    /* Model arguments */
    string path_shape_input;
    string path_value_input;
    string model_arg = "transformer";
    //string model_arg = "bert-base";
    //string model_arg = "bert-large";
    //string model_arg = "gpt2";
    //string model_arg = "t5-small";
    //string model_arg = "t5-base";
    bool model_interval = false;

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
                << "\t-l: 0 [none] / 1 [print execution time]"
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
        else if (arg_str== "-e")
            model_interval = stoi(argv[i+1]);

        else
        {
            std::cerr << "Error: invalid option provided!" << std::endl;
            assert(0);
            exit(1);
        }

        i+=2;
    }

    if(model_interval)
    {
        std::cout << "Running for Execution Time Accumulation with Release Mode" << std::endl;
    }
    interval_init();
    auto start_t = clock_::now();

    /* Init models */
    TopModel<data_t>* model = nullptr;
    std::cout << "Load model parameters..." << std::endl;
    if (model_arg=="transformer") 
    { 
        path_shape_input = "../sentence/shape_128.input";
        path_value_input = "../sentence/value_transformer.input";
        string path_shape = "../params/shape_transformer.param";
        string path_value = "../params/value_transformer.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
        model = new AttentionTransformer<data_t>(param_map["src_embed.0.lut.weight"].pshape[0], param_map["tgt_embed.0.lut.weight"].pshape[0], model_arg);
    }
    else if (model_arg=="bert-base")
    {
        path_shape_input = "../sentence/shape_128.input";
        path_value_input = "../sentence/value_BERT.input";
        string path_shape = "../params/shape_bert_base_uncased.param";
        string path_value = "../params/value_bert_base_uncased.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
        model = new BERT<data_t>(param_map["embeddings.word_embeddings.weight"].pshape[0], param_map["embeddings.word_embeddings.weight"].pshape[0], model_arg);
    }
    else if (model_arg=="bert-large")
    {
        path_shape_input = "../sentence/shape_128.input";
        path_value_input = "../sentence/value_BERT.input";
        string path_shape = "../params/shape_bert_large_uncased.param";
        string path_value = "../params/value_bert_large_uncased.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
        model = new BERT<data_t>(param_map["embeddings.word_embeddings.weight"].pshape[0], param_map["embeddings.word_embeddings.weight"].pshape[0], model_arg);
    }
    else if (model_arg=="gpt2")
    {
        path_shape_input = "../sentence/shape_GPT2.input";
        path_value_input = "../sentence/value_GPT2.input";
        string path_shape = "../params/shape_gpt2.param";
        string path_value = "../params/value_gpt2.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
        model = new GPT2<data_t>(param_map["wte.weight"].pshape[0], param_map["wte.weight"].pshape[0], model_arg);
    }
    else if (model_arg=="t5-base")
    {
        path_shape_input = "../sentence/shape_128.input";
        path_value_input = "../sentence/value_T5.input";
        string path_shape = "../params/shape_t5_base.param";
        string path_value = "../params/value_t5_base.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
        model = new T5<data_t>(param_map["encoder.embed_tokens.weight"].pshape[0], param_map["encoder.embed_tokens.weight"].pshape[0], model_arg);
    }
    else if (model_arg=="t5-small")
    {
        path_shape_input = "../sentence/shape_128.input";
        path_value_input = "../sentence/value_T5.input";
        string path_shape = "../params/shape_t5_small.param";
        string path_value = "../params/value_t5_small.param";
        get_param_shape(path_shape, param_map);
        get_param_value(path_value, param_map);
        model = new T5<data_t>(param_map["encoder.embed_tokens.weight"].pshape[0], param_map["encoder.embed_tokens.weight"].pshape[0], model_arg);
    }
    else
    {
        std::cerr << "Error: unavailable models" << std::endl;
        assert(0);
        exit(1);
    }

    auto mid_t = clock_::now();
    auto model_init_t = INTERVAL(start_t);

    std::cout << "Model initiailization completes!" << std::endl;
    param_map.clear( );

    /* Load input */
    get_param_shape(path_shape_input, input_idx);
    get_param_value(path_value_input, input_idx);

    Tensor<data_t> input(input_idx["src_idx"].pvals, input_idx["src_idx"].pshape);
    Tensor<data_t> output;
    std::cout << "Input tensor is loaded dimension: " << input << std::endl;

    /* Run model */
    std::cout << "Run the constructed model" << std::endl;
    model->forward(output, input);

    //delete model;
    std::cout << "done" << std::endl;

    auto total_t = INTERVAL(start_t);
    auto model_fwd_t = INTERVAL(mid_t);

    if(model_interval)
    {
        map<string, std::chrono::nanoseconds>::iterator iter_latency;
        std::cout << "Print out Procedure Execution Time" << std::endl;
        for(iter_latency = interval_map.begin(); iter_latency != interval_map.end(); ++iter_latency)
        {
            std::cout << iter_latency->first << " : "
            << chrono::duration_cast<chrono::microseconds>(iter_latency->second).count()
            << "us" << std::endl;
        }
        std::cout << std::endl;
        std::cout << ">>>>>>>> Model Execution Time Summation" << std::endl;
        std::cout << "Model Initialize Execution Time : "
        << chrono::duration_cast<chrono::microseconds>(model_init_t).count()
        << "us" << std::endl;

        std::cout << "Model Forward Execution Time : "
        << chrono::duration_cast<chrono::microseconds>(model_fwd_t).count()
        << "us" << std::endl;

        std::cout << "Total Model Execution Time : "
        << chrono::duration_cast<chrono::microseconds>(total_t).count()
        << "us" << std::endl;
    }
    return 0;
}

