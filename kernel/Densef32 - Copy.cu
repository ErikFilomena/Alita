//Single precision CUDA kernels
#include "Densef32.h"
#include "CommonOp.cuh"
#include <omp.h>

/*
* KERNEL FUNCTIONS
* 
* 
* 
*/

//Strided Two dimensional Kernel, the thread.x represent the row and thread.y the column in the matrix; batchSize is the stride
__global__
void cuda_DenseWeightGradf32(float* weightsGrad, float* input, float* outputGrad, unsigned int inputSize,
	unsigned int outputSize, unsigned int batchSize) {

	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	
	if (i < inputSize && j < outputSize) {
		int t = 0;
		int index = j * inputSize + i;
		input = input + i;
		outputGrad = outputGrad + j;
		float val = 0;
		while (t < batchSize) {
			val += input[0] * outputGrad[0];
			input += inputSize;
			outputGrad += outputSize;
			t++;
		}
		weightsGrad[index] += val;
	}
}




//Input gradient kernel. The x dimension is the input size the y dimension is the batch
__global__ 
void cuda_DenseInputGradf32(float* inputGrad,float* weights,float* outputGrad, unsigned int inputSize, unsigned int outputSize
	,unsigned int batchSize) {

	int inputIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int batchIndex = threadIdx.y + blockIdx.y * blockDim.y;


	extern __shared__ float sharedOutputGrad[]; // Size of output

	if (batchIndex < batchSize) {
		outputGrad += batchIndex * outputSize;
		int sharedIndex = inputIndex;
		while (sharedIndex < outputSize) {
			sharedOutputGrad[sharedIndex] = outputGrad[sharedIndex];
			//printf("%f\n", sharedOutputGrad[sharedIndex]);
			sharedIndex += gridDim.x * blockDim.x;
		}
		__syncthreads();

		if (inputIndex < inputSize) {
			int iptOff = batchIndex * inputSize + inputIndex;
			float val = 0;
			for (int j = 0; j < outputSize; j++) {
				val += weights[j * inputSize + inputIndex] * sharedOutputGrad[j];
			}
			inputGrad[iptOff] += val;
		}

	}
	

	
	
	
}


/***
* Host functions
* Convolution2Df32
* Convolution2DWeightsGradf32
*
***/



cudaError_t DenseWeightGradf32(
	float* weights,float* weightsGrad, float* input,float* inputGrad,
	float* output, float* outputGrad, float* biasGrad, unsigned int inputSize,unsigned int outputSize, 
	unsigned int batchSize) {
	
	cudaError_t status;

#pragma omp parallel
	{
		int id = omp_get_thread_num();
		int nthreads = omp_get_num_threads();

		if (id == 0 && inputGrad) {

			cudaStream_t streamInput;
			cudaStreamCreate(&streamInput);
			dim3 blocksInput((inputSize + 15) / 16, (batchSize + 15) / 16);
			dim3 threadsInput(16, 16, 1);
			cuda_DenseInputGradf32 << <blocksInput, threadsInput, outputSize * sizeof(float), streamInput >> > (inputGrad, weights, outputGrad, inputSize, outputSize, batchSize);
			cudaStreamDestroy(streamInput);

		}

		if (id == 1 || (nthreads <=2&&id==0) && biasGrad) {
			unsigned int nBLocksBias = (outputSize * batchSize + 31) / 32;
			cudaStream_t streamBias;
			cudaStreamCreate(&streamBias);
			cuda_VectorAddf32 << <nBLocksBias, 32, 0, streamBias >> > (biasGrad, outputGrad, biasGrad, batchSize * outputSize);
			cudaStreamDestroy(streamBias);
		}

		if (id == 2 || (nthreads <= 2) && (id == 0)) {
			dim3 blocks((inputSize + 15) / 16, (outputSize + 15) / 16);

			dim3 threads(16, 16);
			cudaStream_t streamWeights;
			cudaStreamCreate(&streamWeights);
			cuda_DenseWeightGradf32 << <blocks, threads, 0, streamWeights >> > (weightsGrad, input, outputGrad, inputSize,
				outputSize, batchSize);
			cudaStreamDestroy(streamWeights);
		}
		
	}

	status = cudaDeviceSynchronize();

	return status;
}

