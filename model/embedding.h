#ifndef _EMBEDDING_H_
#define _EMBEDDING_H_

#include <vector>

#include "top_model.h"
#include "tensor.h"
#include "layer.h"

using namespace std;

namespace text_attention 
{
template<typename T>
class Embedding : public Layer<T>
{
public:
    Embedding(string name, int dim_model, Tensor<T>& lut_em, Tensor<T>& lut_pe)
    : name(name), dim_model(dim_model)
    {
        this->lut_em = &lut_em;
        this->lut_pe = &lut_pe;
    }

    uint64_t parameterCount( ) override
    {
        return (lut_em->size( )+lut_pe->size( ));
    }

    ~Embedding( )
    {
        delete lut_em;
        delete lut_pe;
    }

    void print_params( ) override
    {
        std::cout << "Init embedding " << name 
            << " embeddingTable.shape=" << *lut_em
            << " positionalEncoding.shape=" << *lut_pe << std::endl;
    }

    void forward(const Tensor<T>& input, Tensor<T>& output) override
    {
        /* Set shape */
        std::vector<int> out_shape = input.shape;
        out_shape.push_back(dim_model);
        output.reshape(out_shape);

        /* Set value */
        for (int idx=0; idx<input.size( ); idx++)
        {
            for (int ebd=0; ebd<dim_model; ebd++)
            {
                output[idx*dim_model+ebd] = 
                    (*lut_em)[input[idx]*dim_model+ebd] *
                    std::sqrt(dim_model) + (*lut_pe)[idx*dim_model+ebd];
            }
        }
    }

private:
    Tensor<T>* lut_em; // embedding table
    Tensor<T>* lut_pe; // positional encoding table
    std::string name;
    int dim_model;
};
};

#endif
