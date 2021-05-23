#pragma once

#include <stdio.h>
#include "AlitaCore.h"
#include <vector>
#include "Alita.h"
#include "Tensor.h"

class Layerf32 {
public:

	unsigned int inputSize;
	unsigned int outputSize;
	Activation activation;

	Tensor<float,1> input;
	Tensor<float,1> weights;
	Tensor<float,1> bias;
	Tensor<float,1> output;

	float* sumOfWeights;
	float* sumOfOutput;

	float* deviceBuffer;
	float* dataBegin;
	float* dataEnd;
	float* gradient;

	unsigned int lastBatchSize;

	ALITA_CORE_STATUS status;

	Layerf32(unsigned int inputSize, int outputSize, Activation activation) :inputSize(inputSize), outputSize(outputSize), activation(activation) {
		//Computes the size of the layer
		unsigned int size = 2*(2 * inputSize * outputSize + 2 * MAX_BATCH_SIZE * outputSize +
			2 * outputSize + inputSize + MAX_BATCH_SIZE) * (unsigned int)sizeof(float);

		//Allocate memory for the layer
		status.error = cudaMallocManaged(&deviceBuffer, size);

		//Assign the memory to the elements of the layer
		sumOfOutput = deviceBuffer;
		sumOfWeights = deviceBuffer + MAX_BATCH_SIZE;
		size = inputSize * outputSize;
		dataBegin = sumOfWeights + inputSize;

		weights.data = dataBegin;
		weights.size = size;
		bias.data = weights.data + size;
		bias.size = outputSize;
		output.data = bias.data + outputSize;
		output.size = outputSize * MAX_BATCH_SIZE;

		dataEnd = output.data + MAX_BATCH_SIZE * outputSize;

		weights.gradient = dataEnd;
		bias.gradient = dataEnd + size;
		output.gradient = bias.gradient + outputSize;

		

	}

	~Layerf32() {
		if (deviceBuffer) {
			cudaFree(deviceBuffer);
		}
	}

	void Forward(unsigned int batchSize) {
		lastBatchSize = batchSize;
		status = DenseForwardf32(input.data, output.data, weights.data, bias.data, inputSize, outputSize, batchSize, activation);

	}
	void Backward() {
		
		status = DenseGradf32(input.data, input.gradient, inputSize, output.data, output.gradient, outputSize, weights.data, weights.gradient,
			bias.gradient, lastBatchSize, sumOfOutput, sumOfWeights, activation);
	}
};