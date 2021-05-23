#include "GradientDescent.h"

__global__
void cuda_GradientDescentf32(float* weigths, float* weigthsGrad,float alpha, float lambda, unsigned int size) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < size) {
		weigths[index] -= (alpha * weigthsGrad[index] - lambda * weigths[index]);
		weigthsGrad[index] = 0;
	}
}


cudaError_t GradientDescentf32(LayersVector layers,float alpha, float lambda) {
	lambda = 2 * lambda;


	for (auto layer : layers) {


		unsigned int size = layer->inputSize * layer->outputSize;
		unsigned int nBlocksWeights = (size + 32) / 32;
		unsigned int nBlocksBias = (layer->outputSize + 32) / 32;
		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
		cuda_GradientDescentf32 << <nBlocksWeights, 32,0, stream1 >> > (layer->weights.data, layer->weights.gradient,alpha,lambda, size);
		cuda_GradientDescentf32 << <nBlocksBias, 32,0, stream2 >> > (layer->bias.data, layer->bias.gradient, alpha, lambda, layer->outputSize);
		cudaStreamDestroy(stream1);
		cudaStreamDestroy(stream2);
	}

	cudaDeviceSynchronize();
	return cudaGetLastError();
}