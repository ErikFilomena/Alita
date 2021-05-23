#pragma once
//Import cuda libraries
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"


//
#define MAX_BATCH_SIZE 128

//Context structures and functions
typedef struct ALITA_CORE_STATUS {

	const char* errorString;

	cudaError_t error = cudaSuccess;
	cublasStatus_t cublasError = CUBLAS_STATUS_SUCCESS;

}  ALITA_CORE_STATUS;

struct CUDA_HANDLE_LIST{
	cublasHandle_t globalCBHand;
	bool initialized = false;
	curandGenerator_t globalGenerator32;
};


struct CUDA_DEDVICE_INFO{
	cudaDeviceProp prop;
};

extern CUDA_HANDLE_LIST ALITA_CORE_INFO;
extern CUDA_DEDVICE_INFO DEVICE_INFO;


void AlitaCreateContext();
void AlitaDestroyContext();
//END structures and functions