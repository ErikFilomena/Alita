#pragma once
#include "AlitaCore.h"
cudaError_t BinaryCrossEntropy(Tensor<float,1>& output, int* target,float* loss);