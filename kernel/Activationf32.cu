#include "Activationf32.h"
#include "CommonOp.cuh"

//Linear Activation kernel
__global__
void cuda_AddBiasf32(float* target, float* bias, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		target[index] += bias[index % targetSize];
		index += stride;
	}
}

//Relu  Activation kernel
__global__
void cuda_AddBiasf32relu(float* target, float* bias, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		if (bias) {
			if ((target[index] += bias[index % targetSize]) < 0) target[index] = 0;
		}
		else {
			if (target[index] < 0) target[index] = 0;
		}

		index += stride;
	}
}

//Relu Gradient kernel
__global__
void cuda_ReluGradf32(float* target, float* src, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		if (src[index] == 0) {
			target[index] = 0;
		}
		index += stride;
	}
}


//Sigmoid  Activation kernel
__global__
void cuda_AddBiasf32sigmoid(float* target, float* bias, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;

	while (index < size) {
		if (bias) target[index] += bias[index % targetSize];
		target[index] = 1.0 / (1 + exp(-target[index]));
		index += stride;
	}
}

//Sigmoid Gradient kernel
__global__
void cuda_SigmoidGradf32(float* target, float* src, unsigned int targetSize, unsigned int size) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	while (index < size) {
		target[index] *= src[index] * (1 - src[index]);
		index += stride;
	}
}

cudaError_t AddBiasActivatef32(float* target, float* bias, unsigned int targetSize, unsigned int batchSize
	, Activation activation, cudaStream_t stream) {

	unsigned int size = targetSize * batchSize;
	unsigned nblocks = (size + 255) / 256;
	switch (activation)
	{
	case relu:
		cuda_AddBiasf32relu << <nblocks, 256, 0, stream >> > (target, bias, targetSize, targetSize * batchSize);
		break;
	case sigmoid:
		cuda_AddBiasf32sigmoid << <nblocks, 256, 0, stream >> > (target, bias, targetSize, targetSize * batchSize);
		break;
	case linear:
		if(bias)cuda_AddBiasf32 << <nblocks, 256, 0, stream >> > (target, bias, targetSize, targetSize * batchSize);
		break;
	}
	cudaError_t status = cudaGetLastError();
	cudaDeviceSynchronize();
	return status;
}

cudaError_t Gradf32(float* target, float* src, unsigned int targetSize, unsigned int batchSize
	, Activation activation, cudaStream_t stream) {

	unsigned int size = targetSize * batchSize;
	unsigned nblocks = (size + 255) / 256;
	switch (activation)
	{
	case relu:
		cuda_ReluGradf32 << <nblocks, 256,0, stream >> > (target, src, targetSize, targetSize * batchSize);
		break;
	case sigmoid:
		cuda_SigmoidGradf32 << <nblocks, 256, 0, stream >> > (target, src, targetSize, targetSize * batchSize);
		break;
	case linear:
		break;
	}
	cudaDeviceSynchronize();
	cudaError_t status = cudaGetLastError();
	return status;
}

void Activationf32::Forward(float* input, int inputSize, float* inputGrad)
{

	return;
}

void Activationf32::Grad(float* input, int inputSize, float* inputGrad)
{
	return;
}


__global__
void cuda_Expf32(float* target,int size) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < size) {
		target[index] = exp(target[index]);
	}
}
void AlitaOnCuda::Expf32::Forward(float* input, int inputSize, float* inputGrad)
{
	cuda_Expf32 << < (inputSize + 31) / 32, 32 >> > (input,inputSize);
	if (inputGrad)cudaMemset(inputGrad, 0, inputSize * sizeof(float));
	cudaDeviceSynchronize();
}


__global__
void cuda_ExpGradf32(float* inputGrad, float* input, int size) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < size) {
		inputGrad[index] *= exp(input[index]);
	}
}
void AlitaOnCuda::Expf32::Grad(float* input, int inputSize, float* inputGrad)
{
	cuda_ExpGradf32 << < (inputSize + 31) / 32, 32 >> > (input,inputGrad, inputSize);
}
