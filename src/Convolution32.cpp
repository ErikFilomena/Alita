#include "Convolutionf32.h"

/// <summary>
/// Forward convolution for f32 data
/// </summary>
/// <param name="inputDescriptor">Descriptor for the input tensor</param>
/// <param name="input">Input data pointer</param>
/// <param name="kernelDescriptor">Descriptor for the Kernel tensor</param>
/// <param name="kernel">Kernel data pointer</param>
/// <param name="convDesciptor">Descriptor for the convolution operation</param>
/// <param name="algo">Convolution algorithim enum</param>
/// <param name="outputDescriptor">Descriptor for the output tensor</param>
/// <param name="output">Output data pointer</param>
/// <returns></returns>
/// 
cudnnStatus_t ConvolutionForwardf32(
	cudnnTensorDescriptor_t& inputDescriptor,
	float* input,
	cudnnFilterDescriptor_t& kernelDescriptor,
	float* kernel,
	cudnnConvolutionDescriptor_t& convDescriptor,
	cudnnTensorDescriptor_t& outputDescriptor,
	float* output
) 
{
	cudnnStatus_t status;
	float alpha = 1;
	float beta = 0;
	status = cudnnConvolutionForward(ALITA_CORE_INFO.globalCudnnHandle,&alpha,inputDescriptor,input,kernelDescriptor,kernel,
		convDescriptor, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,nullptr,0,&beta,outputDescriptor,output);
	if (status != CUDNN_STATUS_SUCCESS) {
		return status;
	}

	return status;
}