#include "BinaryCrossEntropy.h"

__global__
void BinaryCrossEntropy(float* loss, float* output, float* grad, int* target, unsigned int len) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < len) {

		if (target[index] == 1) {
			atomicAdd(loss, log(output[index]));
			grad[index] = -1.0 / output[index];
		}
		else {
			atomicAdd(loss, log(1.0 - output[index]));
			grad[index] = 1.0 / (1.0 - output[index]);
		}
	}
}

cudaError_t BinaryCrossEntropy(Tensor<float,1>& output, int* target, float* loss) {
	unsigned int nBlocks = (output.size +31)/32;
	BinaryCrossEntropy << <nBlocks, 32 >> > (loss,output.data,output.gradient,target, output.size);

	cudaDeviceSynchronize();

	return cudaGetLastError();
}