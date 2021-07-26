#include "AlitaCore.h"



CUDA_HANDLE_LIST ALITA_CORE_INFO;
CUDA_DEDVICE_INFO DEVICE_INFO;

unsigned int MAX_BATCH_SIZE = 100;

void SetMaxBatch(unsigned int n) {
	MAX_BATCH_SIZE = n;
}


void SetSeed(size_t seed) {
	if (ALITA_CORE_INFO.initialized) {
		curandSetPseudoRandomGeneratorSeed(ALITA_CORE_INFO.globalGenerator32, seed);
	}
}

void AlitaCreateContext() {
	cublasCreate(&ALITA_CORE_INFO.globalCBHand);
	ALITA_CORE_INFO.initialized = true;
	cudaGetDevice(&ALITA_CORE_INFO.deviceId);
	cudaGetDeviceProperties(&DEVICE_INFO.prop, ALITA_CORE_INFO.deviceId);
	curandCreateGenerator(&ALITA_CORE_INFO.globalGenerator32,CURAND_RNG_PSEUDO_DEFAULT);
	cudnnCreate(&ALITA_CORE_INFO.globalCudnnHandle);

}
void AlitaDestroyContext() {

	if (ALITA_CORE_INFO.initialized) {
		cublasDestroy(ALITA_CORE_INFO.globalCBHand);
		curandDestroyGenerator(ALITA_CORE_INFO.globalGenerator32);
	}
	cudaDeviceReset();

}



