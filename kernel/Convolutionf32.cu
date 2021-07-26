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


#define idx threadIdx.x
#define idy threadIdx.y
#define idz threadIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z


__global__ 
void cuda_Convolution2Df32(

	float* target, 
	float* image,
	float* kernel,
	float bias,
	int nx,
	int ny,
	int channels,
	int k,
	int batchSize,
	int outputChannels
) 
{

	extern __shared__ float sharedKernel[];
	extern __shared__ float sharedImage[];

	//Index of the first pixel from top to bottom, left to right
	int mi = biy * bdy;
	int mj = bix * bdx;

	//Index in the output
	int i = biy * bdy + idy;
	int j = bix * bdx + idx;
	int z = biz * bdz + idz;
	int kernelChannelStride = k * k;
	int imgSize = nx*ny;

	//Load kernel into shared memory
	int indexBlock = idx + idy * k;
	
	int blockStride = bdx * bdy;
	int kernelSize = k * k * channels;
	
	while (indexBlock < kernelSize) {
		sharedKernel[indexBlock] = kernel[indexBlock];
		indexBlock += blockStride;
	}

	__syncthreads();
	
	//variables to use while loading image into shared memory
	int offset = k / 2;					//The offset w.r.t. the original image
	int yof = mi - offset;				//The relative row
	int xof = mj - offset;				//Relative Col
	int ids = bdy * bdx;				//id stride
	int shy = bdy + k - 1;				//number of shared rows
	int shx = bdx + k - 1;				//number of shared cols
	int shs = shy * shx;				//size in elements of the shared image

	if (z < channels) {
		kernelChannelStride *= z;
		for (int t = 0; t < batchSize; t++) {
			float val = 0;
			//Load Image into shared memory
			int id = idx + idy * bdy;
			while (id < shs) {

				int row = id / shx;
				int col = id - (row * shx);

				if ((yof + row) >= 0 && (yof + row) < ny) {
					if ((xof + col) >= 0 && (xof + col) < nx) {
						sharedImage[id] = image[z * imgSize + (yof + row) * nx + (xof + col)];

					}
					else {
						sharedImage[id] = 0;
					}
				}
				else {
					sharedImage[id] = 0;
				}

				id += ids;
			}
			__syncthreads();
			//END Load Image into shared memory
			if (i < ny && j < nx) {
				for (int row = 0; row < k; row++) {
					for (int col = 0; col < k; col++) {
						val += kernel[col + row * k + kernelChannelStride] * sharedImage[(idy + row) * shx + idx + col];
					}
				}
			}
			if (i < ny && j < nx && z == 0)target[i * nx + j] = bias;
			__syncthreads();
			if (i < ny && j < nx)atomicAdd(&target[i * nx + j], val);
			//printf("%f\n", target[i * nx + j]);
			image += imgSize * channels;
			target += imgSize;
		}
	}
	

}

__global__ //Each thread computes one position of the gradient
void cuda_Convolution2DWeightsGradf32(
	float* kernelGrad, 
	float* image, 
	float* outputGrad,
	int nx, 
	int ny,
	int channels,
	int k,
	int outputChannels,
	int batchSize) 
{

	
	__shared__ float sharedOutputGrad[16][16];
	extern __shared__ float sharedImage[]; // (bdx + k -1)(bdy +k -1)


	int j = idx + bix * bdx;
	int i = idy + biy * bdy;
	int z = biz;

	int imgSize = nx * ny;
	int offset = k / 2;					//The offset w.r.t. the original image
	int yof = biy * bdy - offset;				//The relative row
	int xof = bix * bdx - offset;				//Relative Col
	int ids = bdy * bdx;				//id stride
	int shy = bdy + k - 1;				//number of shared rows
	int shx = bdx + k - 1;				//number of shared cols
	int shs = shy * shx;				//size in elements of the shared image
	if (z < channels) {

		float val = 0;
		int channelOffset = z * imgSize;
		for (int t = 0; t < batchSize; t++) {
			if ( j< nx && i < ny) {
				sharedOutputGrad[idy][idx] = outputGrad[i * nx + j];
				//printf("%f\n", outputGrad[i * nx + j]);
			}
			else {
				sharedOutputGrad[idy][idx] =0;
			}

			int id = idx + idy * bdx;
			
			while (id < shs) {

				int row = id / shx;
				int col = id - (row * shx);
				//printf("%d %d %d\n", row,col, id);
				if ((yof + row) >= 0 && (yof + row) < ny) {
					if ((xof + col) >= 0 && (xof + col) < nx) {
						sharedImage[id] =  image[channelOffset + (yof + row) * nx + (xof + col)];
						//printf("%f\n", image[channelOffset + (yof + row) * nx + (xof + col)]);
					}
					else {
						sharedImage[id] = 0;
					}
				}
				else {
					sharedImage[id] = 0;
				}

				
				id += ids;
			}

			__syncthreads();


			if (idx < k && idy < k) {
				for (int row = 0; row < bdy; row++) {
					for (int col = 0; col < bdx; col++) {
						val += sharedImage[(row + idy) * shx + col + idx] * sharedOutputGrad[row][col];
					}
				}
			}


			image += imgSize * channels;
			outputGrad += imgSize;
		}

		if (idx < k && idy < k) {
			atomicAdd(&kernelGrad[(idy + z * k )* k + idx ],val);
		}

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

cudaError_t Convolution2Df32(float* target, float* image, float* kernel, float bias, int nx, int ny,
	int channels, int k,int batchSize, int outputChannels, cudaStream_t stream) {

	int nThreads = 16;
	dim3 blocks((nx + nThreads -1) / nThreads, (ny + nThreads-1) / nThreads, channels);
	dim3 threads(nThreads, nThreads, 1);
	int shared = k * k * channels + (15+k)*(15+k);
	cuda_Convolution2Df32 << <blocks, threads,shared*sizeof(float),stream >> > (target, image, kernel, bias, nx, ny, channels, 
		k,batchSize, outputChannels);
	cudaDeviceSynchronize();
	cudaError_t status = cudaGetLastError();
	return status;


}

cudaError_t Convolution2DWeightsGradf32(float* weigthsGrad, float* image, float* outputGrad,
	int nx, int ny, int channels, int k,int outputChannels,int batchSize, cudaStream_t stream) {


	dim3 blocks((nx + 15) / 16, (ny + 15) / 16, channels);
	dim3 threads(16, 16, 1);
	int shared = (15 + k) * (15 + k);
	cuda_Convolution2DWeightsGradf32 << <blocks, threads,shared*sizeof(float),stream >> > (weigthsGrad, image, outputGrad, nx, ny,
		channels, k,outputChannels,batchSize);
	cudaDeviceSynchronize();
	cudaError_t status = cudaGetLastError();
	return status;
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

