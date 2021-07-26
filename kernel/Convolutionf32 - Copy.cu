#include "Convolutionf32.h"
#include "CommonOp.cuh"

/***
* Kernel functions
* cuda_Convolution2Df32
* cuda_Convolution2DReluf32
* cuda_Convolution2DReluf32
* cuda_Convolution2DWeightsGradReluf32
***/
//
__global__ 
void cuda_Convolution2Df32(float* target, float* image, float* kernel, float* bias, int nx, int ny,
	int channels, int k,int batchSize, int outputChannels) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < nx && j < ny) {
		
		
		int imgSize = nx * ny;
		int kernelSize = k * k;
		int offset = k / 2;

		int xof = i - offset;
		int yof = j - offset;

		int maxCol = yof + k;

		if (maxCol > ny) {
			++maxCol -= ny;
		}
		else {
			++maxCol = k;
		}

		int maxRow = xof + k;
		if (maxRow > nx) {
			++maxRow -= nx;
		}
		else {
			++maxRow = k;
		}
		for (int t = 0; t < batchSize;t++) {
			//printf("%d\n", (size_t)image);
			float val = 0;
			for (int ch = 0; ch < channels; ch++) {
				int chOffset = ch * imgSize;
				int kernelOffset = ch * kernelSize;
				for (int col = max(0, -yof); col < maxCol; col++) {
					int totalOffset = (yof + col) * nx + chOffset + xof;
					int totalKernelOffset = kernelOffset + col * k;
					for (int row = max(0, -xof); row < maxRow; row++) {
						val += image[row + totalOffset] * kernel[row + totalKernelOffset];
						
					}
				}
			}
			
			if (bias) {
				target[i + j * nx] = bias[i + j * nx] + val;
				
			}
			else {
				target[i + j * nx] = val;
	
			}
			image += imgSize * channels;
			target += imgSize * outputChannels;
		}
	}
}

__global__ //Each thread computes one position of the gradient
void cuda_Convolution2DWeightsGradf32(float* weigthsGrad, float* image, float* outputGrad,
	int nx, int ny, int channels, int k, int outputChannels, int batchSize) {
	int ki = threadIdx.x + blockIdx.x * blockDim.x;
	int kj = threadIdx.y + blockIdx.y * blockDim.y;
	int kz = threadIdx.z + blockIdx.z * blockDim.z;

	if (ki < k && kj < k && kz < channels) {
		int imgSize = nx * ny;
		image = image + kz * imgSize;
		int offset = k / 2;
		int infXOff = max(ki - k / 2, 0);
		int infYOff = max(kj - k / 2, 0);
		int supXOff = min(nx, nx + ki - offset);
		int supYOff = min(ny, ny + kj - offset);
		float val = 0;
		for (int t = 0 ; t < batchSize; t++) {
			for (int j = max(infYOff, 0); j < supYOff; j++) {
				for (int i = infXOff; i < supXOff; i++) {
					val += image[i + j * ny] * outputGrad[i + j * ny];
				}
			}
			
			image += imgSize * channels;
			outputGrad += imgSize * outputChannels;
		}
		
		weigthsGrad[ki + k * (kj + k * kz)] += val;

	}
}


__global__ //Each thread computes one position of the gradient
void cuda_Convolution2DInputGradf32(float* inputGrad, float* kernel, float* outputGrad,
	int nx, int ny, int channels, int k, int outputChannels, int batchSize) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (i < nx && j < ny && z < channels) {

		int index = i + j * ny;
		
		int imgSize = nx * ny;
		inputGrad = inputGrad + z * imgSize;
		kernel = kernel + z * k * k;
		int offset = k / 2;

		int xof = i - offset;
		int yof = j - offset;


		int maxCol= yof + k;
		if (maxCol > ny) {
			++maxCol -= ny;
		}else {
			++maxCol = k;
		}

		int maxRow = xof + k;
		if (maxRow > nx) {
			++maxRow -= nx;
		}
		else {
			++maxRow = k;
		}

		for (int col = max(0,-yof); col < maxCol; col++) {
			int colOffset = col * k;
			for (int row = max(0,-xof); row < maxRow; row++) {
				inputGrad[index] += kernel[row + colOffset]*outputGrad[xof + row + k*(yof+col)];
			}
		}
			
	}
}



/***
* Host functions
* Convolution2Df32
* Convolution2DWeightsGradf32
*
***/

cudaError_t Convolution2Df32(float* target, float* image, float* kernel, float* bias, int nx, int ny,
	int channels, int k,int batchSize, int outputChannels, cudaStream_t stream) {

	dim3 blocks((nx + 15) / 16, (ny + 15) / 16, 1);
	dim3 threads(16, 16, 1);

	cuda_Convolution2Df32 << <blocks, threads,0,stream >> > (target, image, kernel, bias, nx, ny, channels, k,batchSize, outputChannels);
	cudaDeviceSynchronize();

	return cudaGetLastError();


}

cudaError_t Convolution2DWeightsGradf32(float* weigthsGrad, float* image, float* outputGrad,
	int nx, int ny, int channels, int k,int outputChannels,int batchSize, cudaStream_t stream) {

	dim3 blocks((k + 15) / 16, (k + 15) / 16, (channels + 3) / 4);
	dim3 threads(16, 16, 4);

	cuda_Convolution2DWeightsGradf32 << <blocks, threads,0,stream >> > (weigthsGrad, image, outputGrad, nx, ny,
		channels, k,outputChannels,batchSize);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

cudaError_t Convolution2DInputGradf32(float* inputGrad, float* kernel, float* outputGrad,
	int nx, int ny, int channels, int k, int outputChannels, int batchSize, cudaStream_t stream) {
	dim3 blocks((k + 15) / 16, (k + 15) / 16, (channels + 3) / 4);
	dim3 threads(16, 16, 4);
	cuda_Convolution2DInputGradf32 << <blocks, threads,0,stream >> > (inputGrad, kernel, outputGrad, nx, ny,
		channels, k,outputChannels,batchSize);
	cudaDeviceSynchronize();
	return cudaGetLastError();
}

