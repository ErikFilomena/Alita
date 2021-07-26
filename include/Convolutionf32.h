#pragma once

#include "AlitaCore.h"



cudaError_t Convolution2Df32(float* target, float* image, float* kernel, float bias, int nx, int ny,
	int channels, int k,int batchSize, int neurons, cudaStream_t = cudaStreamDefault);

cudaError_t Convolution2DWeightsGradf32(float* weigthsGrad, float* image, float* outputGrad,
	int nx, int ny, int channels, int k, int outputChannels, int batchSize, cudaStream_t = cudaStreamDefault);

cudaError_t Convolution2DInputGradf32(float* inputGrad, float* kernel, float* outputGrad,
	int nx, int ny, int channels, int k, int outputChannels, int batchSize, cudaStream_t = cudaStreamDefault);

