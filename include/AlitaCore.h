#pragma once
//Import cuda libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include "curand.h"
#include "cudnn.h"
///////////////////////

//UTILITY LIBRRIES

#include <string>
#include <stdio.h>
#include <vector>
#include <math.h>
#include "Tensor.h"


#define ALITA_ERROR_STRING std::string

enum padding {no,zero,mirror};
enum Activation { relu, sigmoid, linear };
enum Generatorf32 { uniform, normal };



struct Options {
	float rngMean=0;
	float rngSDev=0.01;
	Generatorf32 gen = normal;
};

struct ConvolutionOptions {
	float rngMean = 0;
	float rngSDev = 0.01;
	padding pad = zero;
	Generatorf32 gen = normal;
};

//MAX_BATCH_SIZE Global Variable
extern unsigned int MAX_BATCH_SIZE;

//Set MAX_BATCH_SIZE Global Variable
void SetMaxBatch(unsigned int n);

void SetSeed(size_t seed);

//Context structures and functions
typedef struct ALITA_CORE_STATUS {


	cudaError_t error = cudaSuccess;
	cublasStatus_t cublasError = CUBLAS_STATUS_SUCCESS;
	curandStatus_t curandError = CURAND_STATUS_SUCCESS;
	cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS;
	cudaError_t error2 = cudaSuccess;

}  ALITA_CORE_STATUS;


struct CUDA_HANDLE_LIST{
	cublasHandle_t globalCBHand;
	bool initialized = false;
	curandGenerator_t globalGenerator32;
	int deviceId;
	cudnnHandle_t globalCudnnHandle;
};

struct CUDA_DEDVICE_INFO{
	cudaDeviceProp prop;
};


//GLOBAL structures and functions
extern CUDA_HANDLE_LIST ALITA_CORE_INFO;
extern CUDA_DEDVICE_INFO DEVICE_INFO;


void AlitaCreateContext();
void AlitaDestroyContext();

//END context structures and functions





