#pragma once
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <windows.h>

typedef struct ALITA_CORE_STATUS {

	const char* errorString;

	union
	{
		cudaError_t err;
		cublasStatus_t cbStatus;
	}error;

	
}  ALITA_CORE_STATUS;

struct CUDA_HANDLE_LIST{
	cublasHandle_t globalCBHand;
	bool initialized = false;
};


struct CUDA_DEDVICE_INFO{
	cudaDeviceProp prop;
};

extern CUDA_HANDLE_LIST ALITA_CORE_INFO;
extern CUDA_DEDVICE_INFO DEVICE_INFO;

enum Activation{relu,sigmoid,linear};


cudaError_t AddBiasActivatef32(float* target, float* bias, unsigned int targetSize
	, unsigned int batchSize, Activation activation);

cudaError_t Gradf32(float* target, float* src, unsigned int targetSize, unsigned int batchSize
	, Activation activation);

cudaError_t DenseWeightGradf32(
	float* weights, float* weightsGrad, float* input, float* inputGrad,
	float* output, float* outputGrad, float* biasGrad, unsigned int inputSize, unsigned int outputSize,
	unsigned int batchSize, float* sumOfOutput, float* sumOfWeights);