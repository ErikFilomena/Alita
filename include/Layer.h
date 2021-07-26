#pragma once


#include "AlitaCore.h"
#include "Densef32.h"

extern unsigned int MAX_THREADS;



class Layerf32 {
public:

	unsigned int inputSize;
	unsigned int outputSize;
	Activation activation;

	Tensor<float,1> input;
	Tensor<float,1> weights;
	Tensor<float,1> bias;
	Tensor<float,1> output;


	float* deviceBuffer;
	float* dataBegin;
	float* dataEnd;
	float* gradient;

	unsigned int lastBatchSize;

	ALITA_CORE_STATUS status;

	Layerf32(unsigned int inputSize, int outputSize, Activation activation,Options options);

	void Forward(unsigned int batchSize);

	void Backward();
};

inline Layerf32::Layerf32(unsigned int inputSize, int outputSize, Activation activation, Options options) :inputSize(inputSize),
outputSize(outputSize), activation(activation) {
	//Computes the size of the layer
	unsigned int size =(2 * inputSize * outputSize + 2 * MAX_BATCH_SIZE * outputSize +
		2 * outputSize) *sizeof(float);

	//Allocate memory for the layer
	status.error = cudaMallocManaged(&deviceBuffer, size);

	//Assign the memory to the elements of the layer
	size = inputSize * outputSize;
	dataBegin = deviceBuffer;

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

	curandGenerateNormal(ALITA_CORE_INFO.globalGenerator32, weights.data, weights.size,options.rngMean,options.rngSDev);
	curandGenerateNormal(ALITA_CORE_INFO.globalGenerator32, bias.data,bias.size, options.rngMean, options.rngSDev);
	cudaMemset(bias.data, 0, outputSize);
	cudaDeviceSynchronize();

}

inline void Layerf32::Forward(unsigned int batchSize) {
	lastBatchSize = batchSize;
	status = DenseForwardf32(input.data, output.data, weights.data, bias.data, inputSize, outputSize, batchSize, activation,false);
	cudaMemset(output.gradient, 0, outputSize * batchSize * sizeof(float));
	cudaDeviceSynchronize();
}


inline void Layerf32::Backward() {

	status = DenseGradf32(input.data, input.gradient, inputSize, output.data, output.gradient, outputSize, weights.data, weights.gradient,
		bias.gradient, lastBatchSize, activation);
}