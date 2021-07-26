#pragma once
#include "AlitaCore.h"
#include "Convolutionf32.h"
#include "Activationf32.h"

#define KERNELS std::vector< Tensor<float,3> > 

class Conv2dLayerf32 {
public:
	int neurons;
	int kernelSize;

	KERNELS kernels;
	
	Tensor<float, 1> input;
	Tensor<float, 1> output;
	Tensor<float, 1> bias;



	Shape<3> inputShape;
	Shape<3> outputShape;

	float* data;
	size_t dataSize;

	std::vector< ALITA_CORE_STATUS > status;

	int imgSize;
	int inputChannels;
	int batchStride;
	int kernelStride;
	int dataStride;

	int lastBatchSize;

	//Image descriptor w, h and ch
	Conv2dLayerf32(int neurons, int kernelSize, Shape<3>& imageDescriptor, Options options);

	~Conv2dLayerf32() { if (data) cudaFree(data); }

	void Forward(unsigned int batchSize, Activation activation = linear);

	void cudnnForward(unsigned int batchSize, Activation activation = linear);

	void Backward(Activation activation = linear);

};

class cudnnConv2dLayer {
public:
	float* data;

	Tensor<float, 1>kernel;
	Tensor<float, 1>input;
	Tensor<float, 1>output;
	Tensor<float, 1> bias;
	

	cudnnTensorDescriptor_t inputDescriptor;
	cudnnTensorDescriptor_t outputDescriptor;
	cudnnFilterDescriptor_t kernelDescriptor;
	cudnnConvolutionDescriptor_t convDescriptor;

	int inptW;
	int inptH;
	int inptCh;

	int otptW;
	int otptH;
	int neurons;

	int k;

	int lastBatchSize;

	int kernelStride; 
	int outputStride;
	int dataStride;

	ALITA_CORE_STATUS status;

	cudnnConv2dLayer(int neurons, int kernelSize, Shape<3>& imageDescriptor, ConvolutionOptions options);

	void Forward(int batchSize, Activation activation);

};