#pragma once

#include "AlitaCore.h"

enum Activation { relu, sigmoid, linear };


ALITA_CORE_STATUS DenseForwardf32(float* input, float* output, float* weights, float* bias, unsigned int inputSize,
	unsigned int outputSize, unsigned int batchSize, Activation activation);
ALITA_CORE_STATUS DenseGradf32(float* input, float* inputGrad, unsigned int inputSize,
	float* output, float* outputGrad, unsigned int outputSize,
	float* weights, float* weightsGrad, float* biasGrad, unsigned int batchSize,
	float* sumOfOutput, float* sumOfWeights, Activation activation);

cudaError_t AddBiasActivatef32(float* target, float* bias, unsigned int targetSize,
	unsigned int batchSize, Activation activation);

cudaError_t Gradf32(float* target, float* src, unsigned int targetSize, unsigned int batchSize,
	Activation activation);

cudaError_t DenseWeightGradf32(
	float* weights, float* weightsGrad, float* input, float* inputGrad,
	float* output, float* outputGrad, float* biasGrad, unsigned int inputSize, unsigned int outputSize,
	unsigned int batchSize, float* sumOfOutput, float* sumOfWeights);


