#include "CommonOp.cuh"
#include "AlitaCore.h"

#include <math.h>

#define min(a,b) a<b?a:b

__global__
void cuda_AddBiasf32(float* target, float* bias,unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		target[index] += bias[index % targetSize];
		index += stride;
	}
}

__global__
void cuda_AddBiasf32relu(float* target, float* bias, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		if( (target[index] += bias[index % targetSize]) < 0) target[index] = 0;
		index += stride;
	}
}

__global__
void cuda_ReluGradf32(float* target, float* src, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		if (src[index] > 0) {
			target[index] *= 1;
		}
		else {
			target[index] = 0;
		}
		index += stride;
	}
}


__global__
void cuda_AddBiasf32sigmoid(float* target, float* bias, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		target[index] += bias[index % targetSize];
		target[index] = 1.0 / (1 + exp(-target[index]));
		index += stride;
	}
}

__global__
void cuda_SigmoidGradf32(float* target, float* src, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		target[index] = src[index] * (1 - src[index]);
		index += stride;
	}
}

__global__
void cuda_DenseWeightGradf32(float* weightsGrad, float* input, float* outputGrad, unsigned int inputSize,
	unsigned int outputSize, unsigned int batchSize) {

	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int t = 0;
	if (i < inputSize && j < outputSize) {
		while (t < batchSize) {
			weightsGrad[j * inputSize + i] += input[i] * outputGrad[j];
			input += inputSize;
			outputGrad += outputSize;
			t++;
		}

	}
}

__global__
void cuda_DenseSumOfOutput(float* sumOfOutput, float* outputGrad, unsigned int outputSize, unsigned int batchSize) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < batchSize) {
		outputGrad += index * outputSize;
		sumOfOutput[index] = 0;
		for (int i = 0; i < outputSize; i++) {
			sumOfOutput[index] += outputGrad[i];
		}
	}
}

__global__
void cuda_DenseSumOfWeights(float* sumOfWeights, float* weights, unsigned int inputSize, unsigned int outputSize) {
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < inputSize) {
		sumOfWeights[index] = 0;
		for (int j = 0; j < outputSize; j++) {
			sumOfWeights[index] += weights[index + j * inputSize];
		}
	}
}

__global__ // 2d Kernel - the x dimension represents the input coordinate and the y dimension the batch position
void cuda_DenseUpdateInputGrad(float* inputGrad, float* sumOfWeights, float* sumOfOutput, unsigned int inputSize,unsigned int batchSize) {
	unsigned int indexInput = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int indexBatch = threadIdx.y + blockDim.y * blockIdx.y;
	if (indexInput < inputSize && indexBatch < batchSize) {
		inputGrad[indexInput + indexBatch * inputSize] += sumOfWeights[indexInput] * sumOfOutput[indexBatch];
	}
}


//Host Code
cudaError_t AddBiasActivatef32(float* target, float* bias, unsigned int targetSize, unsigned int batchSize
	,Activation activation) {
	
	unsigned int size = targetSize * batchSize;
	unsigned int nThreads = DEVICE_INFO.prop.maxThreadsPerBlock;
	unsigned nblocks = min(DEVICE_INFO.prop.multiProcessorCount* nThreads,(nThreads + size -1)/ nThreads);
	switch (activation)
	{
	case relu:
		cuda_AddBiasf32relu << <nblocks, nThreads >> > (target, bias, targetSize, targetSize * batchSize);
		break;
	case sigmoid:
		cuda_AddBiasf32sigmoid << <nblocks, nThreads >> > (target, bias, targetSize, targetSize * batchSize);
		break;
		break;
	case linear:
		cuda_AddBiasf32 << <nblocks, nThreads >> > (target, bias, targetSize, targetSize * batchSize);
		break;
	}
	cudaError_t status = cudaGetLastError();
	cudaDeviceSynchronize();
	return status;
}

cudaError_t Gradf32(float* target, float* src, unsigned int targetSize, unsigned int batchSize
	, Activation activation) {

	unsigned int size = targetSize * batchSize;
	unsigned int nThreads = DEVICE_INFO.prop.maxThreadsPerBlock;
	unsigned nblocks = min(DEVICE_INFO.prop.multiProcessorCount * nThreads, (nThreads + size - 1) / nThreads);
	switch (activation)
	{
	case relu:
		cuda_ReluGradf32 << <nblocks, nThreads >> > (target, src, targetSize, targetSize * batchSize);
		break;
	case sigmoid:
		cuda_SigmoidGradf32 << <nblocks, nThreads >> > (target, src, targetSize, targetSize * batchSize);
		break;
		break;
	case linear:
		break;
	}
	cudaError_t status = cudaGetLastError();
	cudaDeviceSynchronize();
	return status;
}




cudaError_t DenseWeightGradf32(
	float* weights,float* weightsGrad, float* input,float* inputGrad,
	float* output, float* outputGrad, float* biasGrad, unsigned int inputSize,unsigned int outputSize, 
	unsigned int batchSize, float* sumOfOutput,float*sumOfWeights) {

	dim3 blocks((inputSize + 31)/32, (outputSize + 31) / 32);
	
	dim3 threads(32, 32);

	cudaStream_t streamWeights;
	cudaStreamCreate(&streamWeights);
	
	cuda_DenseWeightGradf32 << <blocks, threads,0, streamWeights >> > (weightsGrad, input, outputGrad, inputSize,
		outputSize, batchSize);

	if (biasGrad) {
		unsigned int nBLocksBias = (outputSize * batchSize + 31) / 32;
		cudaStream_t streamBias;
		cudaStreamCreate(&streamBias);
		cuda_VectorAddf32 << <nBLocksBias,32,0,streamBias >> > (biasGrad, biasGrad, outputGrad, batchSize * outputSize);
		cudaStreamDestroy(streamBias);
	}
	


	if (inputGrad) {
		cudaStream_t stream2, stream3, stream4;
		cudaStreamCreate(&stream2);
		cudaStreamCreate(&stream3);
		cudaStreamCreate(&stream4);

		cuda_DenseSumOfWeights << < (inputSize + 31) / 32, 32, 0, stream2 >> > (sumOfWeights, weights, inputSize, outputSize);
		cuda_DenseSumOfOutput << < (batchSize + 31) / 32, 32, 0, stream3 >> > (sumOfOutput, outputGrad, outputSize, batchSize);

		cudaStreamSynchronize(stream2);
		cudaStreamSynchronize(stream3);

		dim3 blocksInput((inputSize + 31) / 32, (batchSize + 31) / 31);
		dim3 threadsInput(32, 32, 1);

		cuda_DenseUpdateInputGrad << <blocksInput, threadsInput, 0, stream4 >> > (inputGrad, sumOfWeights, sumOfOutput, inputSize, batchSize);

		cudaStreamDestroy(stream2);
		cudaStreamDestroy(stream3);
		cudaStreamDestroy(stream4);

		
	}
	
	cudaStreamDestroy(streamWeights);
	
	cudaDeviceSynchronize();

	return cudaGetLastError();
}