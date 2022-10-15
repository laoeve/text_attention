//
// Created by dianh on 2021/04/16.
//

#ifndef ATTENTION_TRANSFORMER_CPP_TENSOR_H
#define ATTENTION_TRANSFORMER_CPP_TENSOR_H

#include <bits/stdc++.h>

namespace text_attention {
typedef std::vector<int> TensorShape;

template<typename T>
class Tensor : public std::vector<T> {
public:
    Tensor() : std::vector<T>() {
    }

    Tensor(std::vector<T>& values, std::vector<int>& shape) {
        this->insert(this->end( ), values.begin( ), values.end( ));
        this->shape = shape;
    }

    Tensor(int size, T default_data) : std::vector<T>(size, default_data) {
        shape.clear();
        shape.push_back(size);
    }

    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &vec) {
        int cnt = 1;
        os << "[" << vec.size() << "]";
        os << "[";
        for (auto i = 0; i < vec.shape.size(); ++i) {
            if (i)
                os << ", ";
            os << vec.shape[i];
            cnt *= vec.shape[i];
        }
        os << "]";
        assert(cnt == vec.size());
        return os;
    }
    
    TensorShape shape;
};
}

#endif //ATTENTION_TRANSFORMER_CPP_TENSOR_H
