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
    Tensor() : std::vector<T>() 
    {
    }

    Tensor(const Tensor<T>& in_tensor)
    {
        *this = in_tensor;
    }

    Tensor(const std::vector<T>& values, const std::vector<int>& shape)
    : shape(shape)
    {
        uint64_t mult = 1;
        for (int i=0; i<shape.size( );i++)
            mult *= shape[i];
        this->resize(mult);

        if (this->size( )!=values.end( )-values.begin( ))
        {
            std::cerr << "Error: shape size does not match "
                "the number of inserting elements" << std::endl;
            assert(0);
            exit(1);
        }
        else
            this->assign(values.begin( ), values.end( ));
    }

    Tensor(const std::vector<int> shape_,
            typename std::vector<T>::iterator firstIt, 
            typename std::vector<T>::iterator lastIt)
    : shape(shape_)
    {
        uint64_t mult = 1;
        for (int i=0; i<shape.size( );i++)
            mult *= shape[i];
        this->resize(mult);
        
        if (this->size( )!=lastIt-firstIt)
        {
            std::cerr << "Error: shape size does not match "
                "the number of inserting elements" << std::endl;
            assert(0);
            exit(1);
        }
        else
            this->assign(firstIt, lastIt);
    }

    Tensor(const std::vector<int> shape_) : shape(shape_)
    {
        uint64_t mult = 1;
        for (int i=0; i<shape.size( );i++)
            mult *= shape[i];
        this->resize(mult);
    }

    Tensor(int size, T default_data) : std::vector<T>(size, default_data) 
    {
        shape.clear();
        shape.push_back(size);
    }

    void reshape(const std::vector<int> shape_,
            typename std::vector<T>::iterator firstIt, 
            typename std::vector<T>::iterator lastIt) 
    {
        shape = shape_;
        uint64_t mult = 1;
        for (int i=0; i<shape.size( );i++)
            mult *= shape[i];
        this->resize(mult);
        
        if (this->size( )!=lastIt-firstIt)
        {
            std::cerr << "Error: shape size does not match "
                "the number of inserting elements" << std::endl;
            assert(0);
            exit(1);
        }
        else
            this->assign(firstIt, lastIt);
    }

    void reshape(const std::vector<int> new_shape)
    {
        shape = new_shape;
        uint64_t mult = 1;
        for (int i=0; i<new_shape.size( );i++)
            mult *= new_shape[i];
        this->resize(mult);
    }

    void reset( )
    {
        this->clear( );
        shape.clear( );
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
        else if (shape.size( )==3)
        {
            /* Batch transpose (1st dimension is batch size) */
            std::vector<T>& vec = *this;
            std::vector<T> tmp_v(shape[0]*shape[1]*shape[2]);

            for (int n=0; n<shape[0]; n++)
            {
                for (int i=0; i<shape[1]; i++)
                {
                    for (int j=0; j<shape[2]; j++)
                        tmp_v[n*shape[1]*shape[2]+j*shape[1]+i] = 
                            vec[n*shape[1]*shape[2]+i*shape[2]+j];
                }
            }

            for (int i=0; i<shape[0]*shape[1]*shape[2]; i++)
                vec[i] = tmp_v[i];

            int tmp_s = shape[1];
            shape[1] = shape[2];
            shape[2] = tmp_s;
        }
        else 
        {
            std::cerr << "Error: only support 2D tensor transpose" << std::endl;
            assert(0);
            exit(1);
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &vec) 
    {
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

        if (cnt!=vec.size( ))
        {
            std::cerr << "Error: shape size does not match "
                << "the number of elements" << std::endl; 
            assert(0);
            exit(1);
        }

//        os << "[";
//        for (int i=0; i<vec.size( ); i++)
//            os << vec[i] << " ";
//        os << "]";

        return os;
    }

    void print_all( ) const
    {
        int cnt = 1;
        std::cout << "[";
        for (int i = 0; i < shape.size(); ++i) {
            if (i)
                std::cout << ", ";
            std::cout << shape[i];
            cnt *= shape[i];
        }
        std::cout << "] ";

        if (cnt!=this->size( ))
        {
            std::cerr << "Error: shape size does not match "
                << "the number of elements" << std::endl; 
            assert(0);
            exit(1);
        }

        std::cout << "[";
        for (int i=0; i<this->size( ); i++)
            std::cout << this->at(i) << " ";
        std::cout << "]" << std::endl;
    }

    bool is_void( ) const
    {
        bool rv = false;
        if (shape.size( )==0 && this->size( )==0)
            rv = true;
        return rv;
    }

    uint64_t get_dims( ) const
    {
        return shape.size( );
    }

    TensorShape shape;
};
}

#endif //ATTENTION_TRANSFORMER_CPP_TENSOR_H
