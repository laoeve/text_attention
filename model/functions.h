//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_FUNCTIONS_H
#define ATTENTION_TRANSFORMER_CPP_FUNCTIONS_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#include <string>
#include <map>
#include <cassert>
#include "tensor.h"

namespace text_attention 
{
typedef struct _pinfo_t
{
    std::vector<int> pshape;
    std::vector<float> pvals;
} pinfo_t;

int max_len = 128;
std::map<std::string, pinfo_t> input_idx;
std::map<std::string, pinfo_t> param_map;

template<typename T>
T GELU(T x) 
{
    return 0.5 * x * (1 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
    // approximate function provide by https://arxiv.org/pdf/1606.08415.pdf
}

template<typename T>
Tensor <T> *get_relative_distances(int windowSize) 
{
    auto *ret = new Tensor<T>();
    std::vector<std::pair<T, T> > tmp;
    for (int i = 0; i < windowSize; ++i) {
        for (int j = 0; j < windowSize; ++j) {
            tmp.emplace_back((T) i, (T) j);
        }
    }

    for (int i1 = 0; i1 < windowSize; ++i1) {
        for (int j1 = 0; j1 < windowSize; ++j1) {
            for (int i2 = 0; i2 < windowSize; ++i2) {
                for (int j2 = 0; j2 < windowSize; ++j2) {
                    ret->push_back(tmp[i2 * windowSize + j2].first - tmp[i1 * windowSize + j1].first);
                    ret->push_back(tmp[i2 * windowSize + j2].second - tmp[i1 * windowSize + j1].second);
                }
            }
        }
    }
    ret->shape = (
        windowSize * windowSize,
        windowSize * windowSize,
        2
    );
    return ret;
}

template<typename T>
Tensor <T> *create_mask(int windowSize, int displacement, 
        bool upperLower, bool leftRight) 
{
    int sizeV2 = windowSize * windowSize;
    auto *ret = new Tensor<T>(sizeV2 * sizeV2, 0);
    ret->shape = (sizeV2, sizeV2);
    if (upperLower) {
        for (int pos = 0; pos < sizeV2 * sizeV2; ++pos) {
            int d1 = pos / sizeV2;
            int d2 = pos % sizeV2;
            if ((d1 >= sizeV2 - displacement * windowSize and 
                 d2 < sizeV2 - displacement * windowSize) or
                (d1 < sizeV2 - displacement * windowSize and 
                 d2 >= sizeV2 - displacement * windowSize)) {
                (*ret)[pos] = -INFINITY;
            }
        }
    }
    if (leftRight) {
        for (int pos = 0; pos < sizeV2 * sizeV2; ++pos) {
            int tmp = pos;
            int d4 = tmp % windowSize;
            tmp /= sizeV2;
            int d2 = tmp % windowSize;
            if ((d2 >= windowSize - displacement and d4 < windowSize - displacement) or
                (d2 < windowSize - displacement and d4 >= windowSize - displacement)) {
                (*ret)[pos] = -INFINITY;
            }
        }
    }
    return ret;
}

/*     std::map<int, std::vector<float> > sentence_parsing(std::string filename) {
    std::cout << "Vocabulary Parsing : " << filename << std::endl;
    std::ifstream input;
    input.open(filename);
    const bool print_log = false;
    const std::regex re(", ");
    
    std::string line;
    std::map<int, std::vector<float> > sentence;
    std::vector<float> tensor_return;
    int cell_num=0;
    int sentence_num=0;

    while(input.peek() != EOF) {
        std::getline(input, line);
        std::vector<std::string> tokenized(
        std::sregex_token_iterator(line.begin(), line.end(), re, -1),
        std::sregex_token_iterator()
        );
        if(print_log == true){
            std::cout << "tokenizing : " << line << " with " << tokenized.size() << std::endl;
            std::cout << tokenized[0] << tokenized[2] << std::endl;
        }
        for(int i=1; i<tokenized.size() ; i++){
            tensor_return.push_back(std::stod(tokenized[i]));
        }

        sentence.insert( std::pair< int, std::vector<float> >(((sentence_num*128)+stoi(tokenized.at(0))), tensor_return) );      
        if(stoi(tokenized.at(0)) == 127){
            sentence_num += 1;
        }
        tensor_return.clear();
    }

    input.close();
    return sentence;
}   */
//    std::map<int, std::vector<float> > sentence_embed = text_attention::sentence_parsing("../sentence.csv");

/*     template<typename T>
void tensor_print(Tensor <T> &input){
    std::cout << "Tensor Shape : ";
    for(int i=0; i < input.shape.size() ; ++i){std::cout << "[" << input.shape[i] << "]"};
    std::cout << std::endl;

    std::cout << "Tensor Element : ";
    for(int i=0; i < input.shape.size() ; ++i){
        std::cout << "Dim" << i+1 << " : "
        for(int j=0; j < input.shape[i] ; ++j){
            std::cout << input[i*input.shape[i]]
        }
    }
} */

template<typename T>
void embed_idx(const Tensor <T> &input, Tensor <T> &output, 
        int dim_model, std::string str_lut, std::string str_pe)
{
    std::cout << "Embedding : " << input << std::endl;

    std::vector<float> lut;
    
    // Embeddings, PE by reference input[idx]
    for(int idx = 0; idx < input.size() ; idx++){
        for(int embed = 0; embed < dim_model; embed++){
            lut.push_back( text_attention::param_map[str_lut].pvals[input[idx]*dim_model + embed] * std::sqrt(dim_model) //Embeddings
                    + text_attention::param_map[str_pe].pvals[input[idx]*dim_model + embed] );  // Positional Encoding
        }
    }
    output.clear();
    output.shape.clear();
    output.insert(output.begin(), lut.begin(), lut.end());
    output.shape = input.shape;
    output.shape.push_back(dim_model);
    lut.clear();
}

template<typename T>
void t_tran(Tensor <T> &input)
{
    int input_whole_size = 1;
    for (auto i = 0; i < input.shape.size(); ++i) {
        input_whole_size *= input.shape[i];
    }
    assert(input.size() == input_whole_size);

    text_attention::Tensor<T> tmp(input_whole_size,0);
    
    for(int row = 0 ; row < input.shape[input.shape.size()-2] ; ++row){
        for(int col = 0 ; col < input.shape[input.shape.size()-1] ; ++col){
            tmp[row + col*(input.shape[input.shape.size()-2])] = input[(input.shape[input.shape.size()-1])*row + col];
        }
    }
    tmp.shape.clear();
    tmp.shape.insert(tmp.shape.end(), input.shape.begin(), input.shape.end());
    std::cout << tmp << std::endl;
    input.clear();
    input.shape.clear();
    
    input.insert(input.end(), tmp.begin(), tmp.end());
    input.shape.insert(input.shape.end(), tmp.shape.begin(), tmp.shape.end());
    std::swap(input.shape[input.shape.size()-1], input.shape[input.shape.size()-2]);

    tmp.clear();
    tmp.shape.clear();
}

std::map<int, std::string> vocab_parsing(std::string filename) 
{
    std::cout << "!Vocabulary Parsing : " << filename << "!" << std::endl;
    std::ifstream input;
    input.open(filename);
    const bool print_log = false;
    const std::regex re("\\('(.*?)', ([0-9]*)\\)");
    
    std::string line;
    std::map<int, std::string> vocab;

    while(input) {
        getline(input, line);
        int vocab_num=0;
        std::match_results<std::string::const_iterator> m;
        while(regex_search(line, m, re)){
            vocab.insert(std::pair<int, std::string>(stoi(m[2]), m[1]));
            vocab_num+=1;
            if(print_log == true){
                for(size_t i=0; i < m.size(); i++){
                    std::cout << m[i] << " at " << vocab_num << std::endl;
                }
            }
            //std::cout << std::endl;
            line = m.suffix();
        }
        
    }
    input.close();
    return vocab;
}

void get_param_shape(std::string fpath, std::map<std::string, pinfo_t> &target_map)
{
    /* Setup parameter size info file */
    std::ifstream fstream;
    fstream.open(fpath.c_str( ));
    if (fstream.is_open()==false)
    {
        std::cerr << "[Error] Failed to open trace file" << std::endl;
        exit(1);
    }

    /* Get parameter info line-by-line */
    std::string fline;
    getline(fstream, fline); // get header
    while (getline(fstream, fline))
    {
        if (fline.empty( ))
            continue;

        /* Get name */
        std::string pname;
        std::istringstream linestream;
        linestream.str(fline);
        linestream >> pname;

        /* Iterate to extract size */
        std::string pshape = "";
        while (linestream.eof( )==0)
        {
            std::string tmp_str;
            linestream >> tmp_str;
            pshape += tmp_str;
        }

        /* Extract size number by eliminating wrap & comma */
        const std::string whead = "torch.Size([";
        const std::string wtail = "])";
        int pos = pshape.find(whead);
        pshape = pshape.replace(pos, whead.length(), "");
        pos = pshape.find(wtail);
        pshape = pshape.replace(pos, wtail.length(), "");

        linestream.str(pshape);
        linestream.clear( ); // reset iostate flag
        std::vector<int> svec;
        std::string stmp;
        while(getline(linestream, stmp, ','))
        {
            svec.push_back(stoi(stmp));
        }
        assert(linestream.eof());

        /* Insert parameters */
        target_map[pname].pshape = svec;
    }

    fstream.close( );
}

void get_param_value(std::string fpath, std::map<std::string, pinfo_t> &target_map)
{
    std::ifstream fstream;
    std::string pname;
    std::string pvals;

    /* Setup parameter file */ 
    fstream.open(fpath.c_str( ));
    if (fstream.is_open()==false)
    {
        std::cerr << "[Error] Failed to open trace file" << std::endl;
        exit(1);
    }

    /* Get parameter info line-by-line */
    std::string fline;
    while (getline(fstream, fline))
    {
        if (fline.empty( ))
            continue;

        /* Check name */
        std::string tmp_name;
        std::istringstream linestream;
        linestream.str(fline);
        linestream >> tmp_name; // get with stream to eliminate whitespace

        if (target_map.count(tmp_name)==1)
        {
            pname = tmp_name;
//            cout << "Start getting parameters of: " << pname << endl;
            continue;
        }

        /* Get weight params */
        linestream.str(fline);  // reset stream string
        linestream.clear( );    // reset iostate flag
        std::string tmp_str;
        while (linestream.eof( )==0)
        {
            linestream >> tmp_str;
            pvals += tmp_str;
        }

        /* Check EOT (end-of-tensor) */
        if (tmp_str.find(")")==std::string::npos)
            continue;

        /* Extract values */
        const std::string whead = "tensor(";
        const std::string wtail = ")";
        const std::string brk_head = "[";
        const std::string brk_tail = "]";

        int pos = pvals.find(whead);
        pvals = pvals.replace(pos, whead.length(), "");
        pos = pvals.find(wtail);
        pvals = pvals.replace(pos, wtail.length(), "");

        pos = pvals.find(brk_head);
        while (pos!=std::string::npos)
        {
            pvals = pvals.replace(pos, brk_head.length(), "");
            pos = pvals.find(brk_head);
        }

        pos = pvals.find(brk_tail);
        while (pos!=std::string::npos)
        {
            pvals = pvals.replace(pos, brk_tail.length(), "");
            pos = pvals.find(brk_tail);
        }

//        cout << pvals << endl;

        linestream.str(pvals);
        linestream.clear( ); 
        std::vector<float> vvec;
        std::string vtmp;
        while(getline(linestream, vtmp, ','))
        {
            vvec.push_back(stof(vtmp));
        }
        assert(linestream.eof());

        /* Insert parameters */
        target_map[pname].pvals = vvec;

//        cout << "Finish getting parameters of : " 
//            << pname << " " << target_map[pname].pvals.size( ) << endl;
    
        pname = "";
        pvals = "";
    }
    
    fstream.close( );
}

/*     std::vector<std::string> find_map_key(std::string keyword, std::map<std::string, pinfo_t> &target_map){
            std::vector<std::string> plist;
    for(std::map<std::string, pinfo_t>::iterator it=target_map.begin(); it != target_map.end() ; it++){
        if(it->first.find(keyword) != std::string::npos)
            plist.push_back(it->first);
    }
    return plist;
}

map<std::string, pinfo_t> find_key_iter(std::string prefix, map<std::string, pinfo_t> &target_map){
    return map<std::string,std::string>::const_iterator m = target_map.lower_bound(prefix);
} */

}

#endif //ATTENTION_TRANSFORMER_CPP_FUNCTIONS_H
