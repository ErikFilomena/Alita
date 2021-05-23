#pragma once
#include "CommonOp.cuh"
#include "AlitaCore.h"

#include "Tensor.h"

cudaError_t BinaryCrossEntropy(Tensor<float,1>& output, int* target);