#include "Conv2dLayer.h"

#include <omp.h>


Conv2dLayerf32::Conv2dLayerf32(int neurons, int kernelSize, Shape<3>& imageDescriptor, Options options)
{
	this->neurons = neurons;
	inputShape = imageDescriptor;
	this->kernelSize = kernelSize;
	outputShape.dims[0] = imageDescriptor.dims[0];
	outputShape.dims[1] = imageDescriptor.dims[1];
	outputShape.dims[2] = neurons;

	imgSize = imageDescriptor.dims[0] * imageDescriptor.dims[1];
	inputChannels = imageDescriptor.dims[2];
	kernelStride = kernelSize * kernelSize * inputChannels;
	status.reserve((neurons + 1));
	for (int i = 0; i < neurons + 1; i++)status.push_back(ALITA_CORE_STATUS());

	dataSize = 2 * neurons * (kernelStride + neurons + imgSize * MAX_BATCH_SIZE);

	status[0].error = cudaMallocManaged(&data, dataSize * sizeof(float));
	cudaDeviceSynchronize();

	kernelStride = kernelSize * kernelSize * inputShape.dims[2];

	dataStride = dataSize / 2;

	for (size_t i = 0; i < neurons; i++) {
		Tensor<float, 3> temp;
		temp.shape.dims[0] = kernelSize;
		temp.shape.dims[1] = kernelSize;
		temp.shape.dims[2] = inputShape.dims[2];

		temp.data = data + i * kernelStride;
		temp.gradient = data + dataStride + i * kernelStride;
		kernels.push_back(temp);
	}
	bias.data = data + neurons * kernelStride;
	output.data = bias.data + neurons;
	bias.gradient = data + dataStride + neurons * kernelStride;
	output.gradient = bias.gradient + neurons;
	output.size = outputShape.dims[0] * outputShape.dims[1];

	status[0].curandError = curandGenerateNormal(ALITA_CORE_INFO.globalGenerator32, data,
		neurons * (kernelStride + kernelStride % 2), options.rngMean, options.rngSDev);

	cudaDeviceSynchronize();

	bias.data[0] = 0;

}

void Conv2dLayerf32::Forward(unsigned int batchSize, Activation activation)
{
	lastBatchSize = batchSize;
	int i;
#pragma omp parallel for
	for (i = 0; i < neurons; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		int stride = imgSize * i;
		status[i + 1].error = Convolution2Df32(output.data + stride, input.data, kernels[i].data,bias.data[i], inputShape.dims[0],
			inputShape.dims[1], inputChannels, kernelSize, batchSize, neurons, stream);
		AddBiasActivatef32(output.data + stride, nullptr, imgSize * inputChannels, batchSize, activation, stream);
		cudaStreamDestroy(stream);
	}
	cudaMemset(output.gradient, 0, imgSize * neurons * batchSize*sizeof(float));
	cudaDeviceSynchronize();
}

void Conv2dLayerf32::Backward(Activation activation)
{
	Gradf32(output.gradient, output.data, imgSize * inputChannels, lastBatchSize, activation);
	int i;
#pragma omp parallel for 
	for (i = 0; i < neurons; i++) {
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		int stride = imgSize * i;
		status[i + 1].error = Convolution2DWeightsGradf32(kernels[i].gradient, input.data, output.gradient, inputShape.dims[0],
			inputShape.dims[1], inputChannels, kernelSize, neurons, lastBatchSize, stream);
		/*if (input.gradient)status[i + 1].error2 = Convolution2DInputGradf32(input.gradient, kernels[i].data, output.gradient, inputShape.dims[0],
			inputShape.dims[1], inputChannels, kernelSize, neurons, lastBatchSize, stream);*/
		cudaStreamDestroy(stream);
	}
	cudaDeviceSynchronize();
}

cudnnConv2dLayer::cudnnConv2dLayer(int neurons, int kernelSize, Shape<3>& imageDescriptor, ConvolutionOptions options)
{
	inptW = imageDescriptor.dims[0];
	inptH = imageDescriptor.dims[1];
	inptCh = imageDescriptor.dims[2];
	this->neurons = neurons;
	k = kernelSize;
	switch (options.pad) {
	case no:
		otptW = inptW - k +1;
		otptH = inptH- k +1;
		break;
	case zero:
		otptW = inptW;
		otptH = inptH;
		break;
	default:
		break;
	}
	
	kernelStride = k * k * inptCh;

	int dataSize = 2 * neurons * (kernelStride + neurons +  otptH*otptW*MAX_BATCH_SIZE) + kernelStride % 2;
	dataStride = dataSize / 2;
	
	status.error = cudaMallocManaged(&data, dataSize);


	kernel.data = data;
	kernel.gradient = data + dataStride;
	bias.data = data + (kernelStride+ kernelStride % 2) * neurons;
	bias.gradient = kernel.gradient + kernelStride * neurons;
	output.data = bias.data + neurons;
	output.gradient = bias.gradient + neurons;
	
	switch (options.gen) {
	case uniform:
		status.curandError = curandGenerateUniform(ALITA_CORE_INFO.globalGenerator32, kernel.data, kernelStride + kernelStride % 2);
		break;
	case normal:
		status.curandError = curandGenerateNormal(ALITA_CORE_INFO.globalGenerator32, kernel.data,
			kernelStride + kernelStride % 2,options.rngMean,options.rngSDev);
		break;
	}

	
	cudnnStatus_t stat = cudnnCreateFilterDescriptor(&kernelDescriptor);
	if (stat != CUDNN_STATUS_SUCCESS) {
		status.cudnnError = stat;
	}
	stat = cudnnCreateTensorDescriptor(&inputDescriptor);
	if (stat != CUDNN_STATUS_SUCCESS) {
		status.cudnnError = stat;
	}
	stat = cudnnCreateTensorDescriptor(&outputDescriptor);
	if (stat != CUDNN_STATUS_SUCCESS) {
		status.cudnnError = stat;
	}
	stat = cudnnCreateConvolutionDescriptor(&convDescriptor);
	if (stat != CUDNN_STATUS_SUCCESS) {
		status.cudnnError = stat;
	}

	stat = cudnnSetFilter4dDescriptor(kernelDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, neurons, inptCh, k, k);
	if (stat != CUDNN_STATUS_SUCCESS) {
		status.cudnnError = stat;
	}

	switch (options.pad) {
	case no:
		stat = cudnnSetConvolution2dDescriptor(convDescriptor, 0, 0,k,k,0,0,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
		if (stat != CUDNN_STATUS_SUCCESS) {
			status.cudnnError = stat;
		}
		break;
	case zero:
		stat = cudnnSetConvolution2dDescriptor(convDescriptor, k/2, k/2, k, k, 0, 0, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
		if (stat != CUDNN_STATUS_SUCCESS) {
			status.cudnnError = stat;
		}
		break;
	default:
		break;
	}

	

}

void cudnnConv2dLayer::Forward(int batchSize, Activation activation)
{
	if (batchSize != lastBatchSize) {
		lastBatchSize = batchSize;
		status.cudnnError = cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
			batchSize, inptCh, inptH, inptW);
		status.cudnnError = cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			batchSize, neurons, otptH, otptW);
	}
	float alpha = 1;
	float beta = 0;
	switch (activation) {
	case linear:
		status.cudnnError = cudnnConvolutionForward(ALITA_CORE_INFO.globalCudnnHandle, &alpha, inputDescriptor, input.data, kernelDescriptor, kernel.data,
			convDescriptor, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, nullptr, 0, &beta, outputDescriptor, output.data);
		break;
	case relu:
		break;
	default:
		break;
	}
	

}
