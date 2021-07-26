#ifndef CUDA_COMMON_OP
#define CUDA_COMMON_OP


__global__
void cuda_VectorAddf32(float* dst, float* x, float* y,unsigned int len);

#endif