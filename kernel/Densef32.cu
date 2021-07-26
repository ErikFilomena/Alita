//Single precision CUDA kernels
#include "Densef32.h"
#include "CommonOp.cuh"


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




//Two dimensional kernel, thread.x represents the row, thread.y represents the column
__global__ 
void cuda_DenseInputGradf32(float* inputGrad,float* weights,float* outputGrad, unsigned int inputSize, unsigned int outputSize
	,unsigned int batchSize) {

	int inputIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int batchIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (inputIndex < inputSize && batchIndex < batchSize) {
		int iptOff = batchIndex * inputSize + inputIndex;
		int optOff = batchIndex * outputSize;
		float val = 0;
		for (int j = 0; j < outputSize; j++) {
			 val += weights[j * inputSize + inputIndex]*outputGrad[optOff +j];
		}
		inputGrad[iptOff] += val;
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

	if (inputGrad) {

		cudaStream_t streamInput;
		cudaStreamCreate(&streamInput);
		dim3 blocksInput((inputSize + 15) / 16, (batchSize + 15) / 16);
		dim3 threadsInput(16, 16, 1);
		cuda_DenseInputGradf32 << <blocksInput, threadsInput, 0, streamInput >> > (inputGrad, weights, outputGrad, inputSize, outputSize, batchSize);
		cudaStreamDestroy(streamInput);
		

#ifdef _DEBUG
		cudaDeviceSynchronize();
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			fprintf(stderr, "Input grad calculation %d", __LINE__);
			return status;
		}
#endif
	}

	if (biasGrad) {
		unsigned int nBLocksBias = (outputSize * batchSize + 31) / 32;
		cudaStream_t streamBias;
		cudaStreamCreate(&streamBias);
		cuda_VectorAddf32 << <nBLocksBias, 32, 0, streamBias >> > (biasGrad, outputGrad, biasGrad, batchSize * outputSize);
		cudaStreamDestroy(streamBias);
	}

#ifdef _DEBUG
	cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr, "Bias grad calculation %d", __LINE__);
		return status;
	}
#endif

	
	dim3 blocks((inputSize + 15)/16, (outputSize + 15) / 16);
	
	dim3 threads(16, 16);
	cudaStream_t streamWeights;
	cudaStreamCreate(&streamWeights);
	cuda_DenseWeightGradf32 << <blocks, threads,0, streamWeights >> > (weightsGrad, input, outputGrad, inputSize,
		outputSize, batchSize);
	cudaStreamDestroy(streamWeights);
	

#ifdef _DEBUG
	cudaDeviceSynchronize();
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		fprintf(stderr,"Error in Dense Weight grad line %d", __LINE__);
		return status;
	}
#endif

	cudaDeviceSynchronize();

	return cudaGetLastError();
}

