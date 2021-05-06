#ifndef CUDA_COMMON_OP
#define CUDA_COMMON_OP

#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"


__global__
void cuda_VectorAddf32(float* dst, float* x, float* y,unsigned int len);

#endif