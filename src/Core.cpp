#include "AlitaCore.h"
#include <omp.h>

#ifdef WIN32
#include <windows.h>

typedef struct {
	float* sumOfOutput;
	float* output;
	unsigned intoutputSize;
	unsigned int batchSize;
}Gradf32ThreadData;



#endif

typedef struct{
	cudaError_t e1;
	cublasStatus_t e2;
}Status;

extern "C" {


ALITA_CORE_STATUS DenseForwardf32(float* input, float* output, float* weights, float* bias, unsigned int inputSize,
	unsigned int outputSize, unsigned int batchSize, Activation activation) {
	
	float alpha = 1;
	float beta = 0;

	
	Status status;
	status.e2 = cublasSgemm(ALITA_CORE_INFO.globalCBHand, CUBLAS_OP_T, CUBLAS_OP_N, outputSize, batchSize, inputSize,
		&alpha, weights, inputSize, input, inputSize, &beta, output, outputSize);
	if (status.e2 != CUBLAS_STATUS_SUCCESS) {
		ALITA_CORE_STATUS temp = { "gemm error" };
		temp.error.cbStatus = status.e2;
		return temp;
	}

	status.e1 = AddBiasActivatef32(output, bias, outputSize, batchSize, activation);
	ALITA_CORE_STATUS temp = { "success" };
	temp.error.err = status.e1;
	return temp;
}

ALITA_CORE_STATUS DenseGradf32(float* input, float* inputGrad, unsigned int inputSize,
	float* output, float* outputGrad, unsigned int outputSize,
	float* weights, float* weightsGrad,float* biasGrad, unsigned int batchSize,
	float* sumOfOutput, float* sumOfWeights, Activation activation) {

	float alpha = 1;
	float beta = 0;

	Status status;

	status.e1 = Gradf32(outputGrad,output,outputSize,batchSize,activation);

	if (status.e1 != cudaSuccess) {
		ALITA_CORE_STATUS temp = { "Gradient error" };
		temp.error.err = status.e1;
		return temp;
	}

	status.e1 = DenseWeightGradf32(weights,weightsGrad,input,inputGrad,output,outputGrad,biasGrad,inputSize,outputSize,
		batchSize,sumOfOutput,sumOfWeights);

	if (status.e1 != cudaSuccess) {
		ALITA_CORE_STATUS temp = { "Gradient error" };
		temp.error.err = status.e1;
		return temp;
	}


	ALITA_CORE_STATUS temp = { "success" };
	temp.error.err = status.e1;
	return temp;
}



}