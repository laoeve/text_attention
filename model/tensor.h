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

    void transpose( ) 
    {
        if (shape.size( )==1) 
        {
            /* Single element */
            if (shape[0]==1)
                return;

            /* Vector */
            int tmp_s = shape[0];
            shape.resize(2);
            shape[0] = 1;
            shape[1] = tmp_s;
        }
        else if (shape.size( )==2)
        {
            /* Vector */
            if (shape[0]==1) 
            {
                int tmp_s = shape[1];
                shape.resize(1);
                shape[0] = tmp_s;
                return;
            }

            /* 2D tensor*/
            std::vector<T>& vec = *this;
            std::vector<T> tmp_v(shape[0]*shape[1]);

            for (int i=0; i<shape[0]; i++)
            {
                for (int j=0; j<shape[1]; j++)
                    tmp_v[j*shape[0]+i] = vec[i*shape[1]+j];
            }

            for (int i=0; i<shape[0]*shape[1]; i++)
                vec[i] = tmp_v[i];

            /* Finalize new shape */
            int tmp_s = shape[0];
            shape[0] = shape[1];
            shape[1] = tmp_s;
        }
        else 
        {
            std::cerr << "Error: only support 2D tensor transpose" << std::endl;
            assert(0);
            exit(1);
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &vec) {
        int cnt = 1;
//        os << "[" << vec.size() << "]";
        os << "[";
        for (int i = 0; i < vec.shape.size(); ++i) {
            if (i)
                os << ", ";
            os << vec.shape[i];
            cnt *= vec.shape[i];
        }
        os << "] ";
        assert(cnt == vec.size());

//        os << "[";
//        for (int i=0; i<vec.size( ); i++)
//            os << vec[i] << " ";
//        os << "]";

        return os;
    }
    
    TensorShape shape;
};
}

#endif //ATTENTION_TRANSFORMER_CPP_TENSOR_H
