#include "AlitaCore.h"


CUDA_HANDLE_LIST ALITA_CORE_INFO;
CUDA_DEDVICE_INFO DEVICE_INFO;

void AlitaCreateContext() {
	cublasCreate(&ALITA_CORE_INFO.globalCBHand);
	ALITA_CORE_INFO.initialized = true;
	int deviceId;
	cudaGetDevice(&deviceId);
	cudaGetDeviceProperties(&DEVICE_INFO.prop, deviceId);
	curandCreateGenerator(&ALITA_CORE_INFO.globalGenerator32,CURAND_RNG_PSEUDO_DEFAULT);
}
void AlitaDestroyContext() {

	if (ALITA_CORE_INFO.initialized) {
		cublasDestroy(ALITA_CORE_INFO.globalCBHand);
		curandDestroyGenerator(ALITA_CORE_INFO.globalGenerator32);
	}

}