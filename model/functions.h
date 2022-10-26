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

std::string replace_str(const std::string input, std::string before, std::string after)
{
    std::string ret = input;
    ret.replace(input.find(before), before.size( ), after);
    return ret;
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

std::map<int, std::string> vocab_parsing(std::string filename) 
{
    std::ifstream input;
    input.open(filename);
    const bool print_log = false;
    const std::regex re("\\(['\"](.*?)[\"'], ([0-9]*)\\)");

    if (input.is_open()==false)
    {
        std::cerr << "[Error] Failed to open trace file" << std::endl;
        assert(0);
        exit(1);
    }

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
        assert(0);
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
            pvals.clear();
//            cout << "Start getting parameters of: " << pname << endl;
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
        const std::string dtype_tail = "dtype";

        int pos;
        while(pvals.find(pname) != std::string::npos)
        {
            pos = pvals.find(pname);
            pvals = pvals.replace(pos, pname.length(), ""); // for duplicated pname
        }
        pos = pvals.find(whead);
        pvals = pvals.replace(pos, whead.length(), "");

        if(pvals.find(dtype_tail) != std::string::npos)
        {
            pos = pvals.find(tmp_str);
            pvals = pvals.replace(pos, tmp_str.length(), ""); // for duplicated pname
        }
        else
        {
        pos = pvals.find(wtail);
        pvals = pvals.replace(pos, wtail.length(), "");
        }

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

        cout << "Finish getting parameters of : " 
            << pname << " " << target_map[pname].pvals.size( ) << endl;
    
        pname = "";
        pvals = "";
    }
    
    fstream.close( );
}

}

#endif //ATTENTION_TRANSFORMER_CPP_FUNCTIONS_H