#include "MemoryManager.h"



void MemoryManager::AllocateDevice(size_t nBytes)
{
	try {
		if (nBytes > maxBufferSize)nBytes = maxBufferSize;
		cudaError_t status = cudaMallocManaged(&deviceBuffer, nBytes);
		cudaDeviceSynchronize();
		if (status != cudaSuccess) {
			throw status;
		}
	}
	catch (cudaError_t& err) {
		printf("Error allocating memory on the device.\n Cuda Error %s", cudaGetErrorString(err));
	}
}

void MemoryManager::AllocateHost(size_t nBytes)
{
	try {
		if (nBytes > maxBufferSize)nBytes = maxBufferSize;
		cudaError_t status = cudaMallocHost(&hostBuffer, nBytes);
		cudaDeviceSynchronize();
		if (status != cudaSuccess) {
			throw status;
		}
	}
	catch (cudaError_t& err) {
		printf("Error allocating memory on the host.\n Cuda Error %s", cudaGetErrorString(err));
	}
}

void MemoryManager::GetDeviceInfo()
{
	int deviceId;
	cudaGetDevice(&deviceId);
	cudaGetDeviceProperties(&deviceInfo,deviceId);
	cudaDeviceSynchronize();
}

void MemoryManager::GetHostInfo()
{
	GetSystemInfo(&hostInfo);
}

MemoryManager::MemoryManager()
{
	int deviceId;
	cudaGetDevice(&deviceId);
	cudaGetDeviceProperties(&deviceInfo, deviceId);
	GetSystemInfo(&hostInfo);
	cudaDeviceSynchronize();
	maxBufferSize = deviceInfo.totalGlobalMem*.8;
}

MemoryManager::~MemoryManager()
{
	if (deviceBuffer) {
		cudaFree(deviceBuffer);
		cudaDeviceSynchronize();
	}
	if (hostBuffer) {
		cudaFree(hostBuffer);
	}
}
