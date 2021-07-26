#include "CommonOp.cuh"


//dst = x + y
__global__
void cuda_VectorAddf32(float *dst, float* x, float* y, unsigned int len) {

	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < len) {
		dst[index] = x[index] + y[index];
	}

}


__global__//dst = beta*dst + alpha*x + lambda*y
void cuda_VectorAlphaBetaAddf32(float* dst, float beta, float* x, float alpha, float* y, float lambda, int size){
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < size) {
		dst[index] = beta * dst[index] + alpha * x[index] + lambda * y[index];
	}
}


__global__//dst = beta*dst + xy' 
void cuda_Outerf32(float* dst, float beta, float* x, int n, float* y, int m) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	if (ix < n && iy < m) {
		dst[ix + iy * n] = x[ix] + y[iy];
	}
}

__global__//dst = dst + sum x_iy_i' 
void cuda_BatchedOuterf32(float* dst, float* x, int n, float* y, int m, int batchSize) {
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int size = n * m;
	if (ix < n && iy < m) {
		for (int t = 0; t < batchSize; t++) {
			dst[ix + iy * n] += x[ix] + y[iy];
			dst += size;
			x += n;
			y += m;
		}
		
	}
}