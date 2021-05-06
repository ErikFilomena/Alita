#pragma once
#include <vector>
#include <stdio.h>
#include <windows.h>

#include "cuda_runtime.h"

struct info {
	void** data;
	size_t nBytes;
	bool* onCuda;
};

class MemoryManager
{
private:
	void* deviceBuffer =nullptr;
	void* hostBuffer = nullptr;
	
	std::vector<info> managed;
	
	size_t maxBufferSize;
public:

	size_t nBytesDevice;
	size_t nBytesHost;

	cudaDeviceProp deviceInfo;
	SYSTEM_INFO hostInfo;

	MemoryManager();
	~MemoryManager();

	void AllocateDevice(size_t nBytes);
	void AllocateHost(size_t nBytes);

	void GetDeviceInfo(); 
	void GetHostInfo();

};

