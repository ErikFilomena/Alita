#include "Softmax.h"


__global__
void cuda_Softmaxf32(float* loss,float* input, float* inputGrad, int* target, int inputSize, int batchSize) {


	extern __shared__ float softmax[];
	
	if (threadIdx.x < inputSize) {
		float localSoftmax = 0;
		if (threadIdx.x == 0) softmax[blockIdx.x] = 0;
		__syncthreads();
		unsigned int index = threadIdx.x + blockIdx.x * inputSize;
		unsigned int pos = threadIdx.x;
		unsigned int stride = blockDim.x;
		while (pos< inputSize) {
			float val = exp(input[index]);
			
			localSoftmax += val;
			input[index] = val;
			pos += stride;
			index += stride;
		}
		
		
		atomicAdd(&softmax[blockIdx.x], localSoftmax);
		__syncthreads();

		
		index = threadIdx.x + blockIdx.x * inputSize;
		pos = threadIdx.x;

		while (pos < inputSize) {
			input[index] /= softmax[blockIdx.x];
			
			if (target[blockIdx.x] == pos) {
				inputGrad[index] =input[index] - 1;
				atomicAdd(loss, -log(input[index]));
			}
			else {
				inputGrad[index] = input[index];
			}
			pos += stride;
			index += stride;
		}


	}
	

}

__global__
void cuda_SoftmaxPredict32(float* input, int* target, int inputSize, int batchSize) {


	extern __shared__ float softmax[];

	if (threadIdx.x < inputSize) {
		float localSoftmax = 0;
		if (threadIdx.x == 0) softmax[blockIdx.x] = 0;
		__syncthreads();
		unsigned int index = threadIdx.x + blockIdx.x * inputSize;
		unsigned int pos = threadIdx.x;
		unsigned int stride = blockDim.x;
		while (pos < inputSize) {
			float val = exp(input[index]);

			localSoftmax += val;
			input[index] = val;
			pos += stride;
			index += stride;
		}


		atomicAdd(&softmax[blockIdx.x], localSoftmax);
		__syncthreads();


		index = threadIdx.x + blockIdx.x * inputSize;
		pos = threadIdx.x;

		while (pos < inputSize) {
			input[index] /= softmax[blockIdx.x];
			pos += stride;
			index += stride;
		}
	}


}


//Host Code

cudaError_t Softmaxf32(float* loss, float* input, float* inputGrad, int* target, int inputSize, int batchSize) {
	
	//Each block will compute softmax for one output
	cuda_Softmaxf32 << <batchSize, 32,batchSize*sizeof(float)>> > (loss,input, inputGrad,target,
		inputSize,batchSize);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t SoftmaxPredictf32(float* input, int* target, int inputSize, int batchSize) {

	//Each block will compute softmax for one output
	cuda_SoftmaxPredict32 << <batchSize, 32, batchSize * sizeof(float) >> > (input, target,inputSize, batchSize);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}