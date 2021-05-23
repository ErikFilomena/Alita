#include "Alita.h"

#include <omp.h>





ALITA_CORE_STATUS DenseForwardf32(float* input, float* output, float* weights, float* bias, unsigned int inputSize,
	unsigned int outputSize, unsigned int batchSize, Activation activation) {
	
	float alpha = 1;
	float beta = 0;

	cudaError_t e1;
	cublasStatus_t e2;
	
	e2 = cublasSgemm(ALITA_CORE_INFO.globalCBHand, CUBLAS_OP_T, CUBLAS_OP_N, outputSize, batchSize, inputSize,
		&alpha, weights, inputSize, input, inputSize, &beta, output, outputSize);

	
	if (e2 != CUBLAS_STATUS_SUCCESS) {
		ALITA_CORE_STATUS temp = { "gemm error" };
		temp.cublasError = e2;
		return temp;
	}

	e1 = AddBiasActivatef32(output, bias, outputSize, batchSize, activation);
	cudaDeviceSynchronize();

	if (e1 != cudaSuccess) {
		ALITA_CORE_STATUS temp = { "Activation kernel error" };
		temp.error = e1;
		return temp;
	}

	ALITA_CORE_STATUS temp = { "success" };
	temp.error = e1;
	return temp;
}

ALITA_CORE_STATUS DenseGradf32(float* input, float* inputGrad, unsigned int inputSize,
	float* output, float* outputGrad, unsigned int outputSize,
	float* weights, float* weightsGrad,float* biasGrad, unsigned int batchSize,
	float* sumOfOutput, float* sumOfWeights, Activation activation) {

	float alpha = 1;
	float beta = 0;

	cudaError_t e1;

	//Compute the gradient w.r.t. the activation
	e1 = Gradf32(outputGrad,output,outputSize,batchSize,activation);

	if (e1 != cudaSuccess) {
		ALITA_CORE_STATUS temp = { "Gradient error" };
		temp.error = e1;
		return temp;
	}

	e1 = DenseWeightGradf32(weights,weightsGrad,input,inputGrad,output,outputGrad,biasGrad,inputSize,outputSize,
		batchSize,sumOfOutput,sumOfWeights);

	if (e1 != cudaSuccess) {
		ALITA_CORE_STATUS temp = { "Gradient error" };
		temp.error = e1;
		return temp;
	}

	cudaDeviceSynchronize();

	ALITA_CORE_STATUS temp = { "success" };
	temp.error = e1;
	return temp;
}



