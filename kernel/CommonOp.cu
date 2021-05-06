#include "CommonOp.cuh"


__global__
void cuda_VectorAddf32(float *dst, float* x, float* y, unsigned int len) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < len) {
		dst[index] = x[index] + y[index];
	}

}