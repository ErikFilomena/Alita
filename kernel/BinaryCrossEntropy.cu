#include "BinaryCrossEntropy.h"


__global__
void BinaryCrossEntropy(float* output, float* grad, int* target, unsigned int len) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < len) {

		if (target[index] == 1) {
			grad[index] = -1.0 / output[index];
		}
		else {
			grad[index] = 1.0 / (1.0 - output[index]);
		}
	}
}

cudaError_t BinaryCrossEntropy(Tensor<float,1>& output, int* target) {
	unsigned int nBlocks = (output.size +31)/32;
	BinaryCrossEntropy << <nBlocks, 32 >> > (output.data,output.gradient,target, output.size);

	cudaDeviceSynchronize();

	return cudaGetLastError();
}