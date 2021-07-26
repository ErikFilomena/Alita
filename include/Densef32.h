#pragma once
#include "AlitaCore.h"

//Dense Feed Forward defined in AlitaCore.cpp
ALITA_CORE_STATUS DenseForwardf32(float* input, float* output, float* weights, float* bias, unsigned int inputSize,
	unsigned int outputSize, unsigned int batchSize, Activation activation, bool synch = true);

ALITA_CORE_STATUS DenseGradf32(float* input, float* inputGrad, unsigned int inputSize,
	float* output, float* outputGrad, unsigned int outputSize,
	float* weights, float* weightsGrad, float* biasGrad, unsigned int batchSize, Activation activation);


cudaError_t DenseWeightGradf32(
	float* weights, float* weightsGrad, float* input, float* inputGrad,
	float* output, float* outputGrad, float* biasGrad, unsigned int inputSize, unsigned int outputSize,
	unsigned int batchSize);