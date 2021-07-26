#pragma once

#include "AlitaCore.h"

struct Activationf32 {
	virtual void Forward(float* input, int inputSize, float* inputGrad =nullptr);
	virtual void Grad(float* input, int inputSize, float* inputGrad);
};


namespace AlitaOnCuda {
struct Expf32:Activationf32{
	virtual void Forward(float* input, int inputSize, float* inputGrad = nullptr);
	virtual void Grad(float* input, int inputSize, float* inputGrad);
};
}





cudaError_t AddBiasActivatef32(float* target, float* bias, unsigned int targetSize,
	unsigned int batchSize, Activation activation, cudaStream_t stream = cudaStreamDefault);

cudaError_t Gradf32(float* target, float* src, unsigned int targetSize, unsigned int batchSize,
	Activation activation, cudaStream_t stream = cudaStreamDefault);