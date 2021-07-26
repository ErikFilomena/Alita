#pragma once

#include "cudnn.h"

template <size_t DIM = 1>
struct Shape{

	int dims[DIM];
	Shape() {
		
	}
	Shape(const int dims[]) {
		for (int i = 0; i < DIM; i++) {
			this->dims[i] = dims[i];
		}
	}
	size_t GetDimensions() {
		return DIM;
	}

	Shape operator=(Shape& src) {
		if (this == &src)return *this;
		for (int i = 0; i < DIM; i++) {
			dims[i] = src.dims[i];
		}
		return *this;
	}

};

template<class NBR, size_t DIM = 1>
class Tensor {
public:


	NBR* data;
	NBR* gradient;
	unsigned int size;
	Shape<DIM> shape;

	Tensor(){}


};
