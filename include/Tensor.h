#pragma once

#include <array>
#include <algorithm>

template <size_t DIM = 1>
struct Shape{
	size_t* shp = new size_t[DIM];

	~Shape() {
		if(shp){
			delete[]  shp;
		}
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
