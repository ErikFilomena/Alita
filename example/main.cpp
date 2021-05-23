#include <stdio.h>
#include <vector>
#include "Alita.h"
#include <time.h>

#include "Layer.h"
#include "Tensor.h"
#include "GradientDescent.h"
#include "BinaryCrossEntropy.h"

using namespace std;

#include<vector>
#include<fstream>




int main() {
	AlitaCreateContext();

	std::fstream file("gisette_train.data");
	std::vector<float> data;
	std::vector<int> labels;
	float* _data;

	float value;
	float mean = 0;
	float var = 0;
	int count = 0;
	while (file >> value) {
		data.push_back(value);
		count++;
	}
	printf("%d\n", count);
	file.close();



	for (int i = 0; i < 5000; i++) {
		mean = 0;
		float var = 0;
		for (int j = 0; j < 6000; j++) {
			mean += data[i + j * 5000];
			var += data[i + j * 5000] * data[i + j * 5000];
		}
		mean /= 6000;
		var /= 6000;
		var -= mean * mean;
		var = sqrt(var);
		for (int j = 0; j < 6000; j++) {
			data[i + j * 5000] = (data[i + j * 5000] - mean);
			if (var)data[i + j * 5000] /= var;
		}
	}



	file.open("gisette_train.labels");
	while (file >> value) {
		labels.push_back(value);
	}


	int* cudaLabels;

	cudaMalloc(&cudaLabels, 6000 * sizeof(int));

	cudaMemcpy(cudaLabels, labels.data(), 6000 * sizeof(int), cudaMemcpyHostToDevice);

	cudaError_t status = cudaMallocManaged(&_data, data.size() * sizeof(float));
	status = cudaMemcpy(_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
	

	Layerf32 layer1(5000, 100, sigmoid);
	Layerf32 layer2(100, 1, sigmoid);

	Tensor<float,1> input;
	input.gradient = nullptr;

	status = cudaMallocManaged(&input.data, 5000* 100 * sizeof(float));

	float* cudaData;

	cudaMallocHost(&cudaData, data.size() * sizeof(float));
	cudaMemcpy(cudaData, data.data(), data.size() * sizeof(float), cudaMemcpyHostToHost);


	curandStatus_t rndstat =curandGenerateNormal(ALITA_CORE_INFO.globalGenerator32, layer1.dataBegin, (size_t)(layer1.dataEnd - layer1.dataBegin) * sizeof(float), 0,.05);
	rndstat =curandGenerateNormal(ALITA_CORE_INFO.globalGenerator32, layer2.dataBegin, (size_t)(layer2.dataEnd - layer2.dataBegin) * sizeof(float), 0, .05);

	layer2.input = layer1.output;
	for (int epoch = 0; epoch < 10; epoch++) {
		double t = clock();
		for (int i = 0; i < 60; i++) {
			//cudaMemcpy(input.data, data.data() + 5000 * i * 100, 5000 * 100 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(input.data, cudaData + 5000 * i * 100, 5000 * 100 * sizeof(float), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			//input.data = cudaData + 5000 * i * 100;
			layer1.input = input;
			layer1.Forward(100);
			layer2.Forward(100);
			cudaDeviceSynchronize();
			BinaryCrossEntropy(layer2.output, cudaLabels + 100 * i);
			layer2.Backward();
			layer1.Backward();
			GradientDescentf32({ &layer1,&layer2 }, 0.00001, 0);
		}
		t = clock() - t;
		
		printf("Epoch in (s): %f\n", t / CLOCKS_PER_SEC);
	}
	
	for (int i = 0; i < 120; i++) {
		cudaMemcpy(input.data, data.data() + 5000 * i * 50, 5000 * 50 * sizeof(float), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		layer1.input = input;
		layer1.Forward(50);
		layer2.Forward(50);
		for (int j = 0; j < 50; j++)printf("%f %d\n", layer2.output.data[j], (labels.data() + i * 50)[j]);
	}

	
	printf("%s\n", cudaGetErrorString(layer1.status.error));

	
	cudaFree(input.data);
	cudaFree(input.gradient);
	cudaFree(cudaLabels);
	AlitaDestroyContext();
	printf("Context destruction succesful");
}