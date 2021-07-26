#include "AlitaCore.h"

cudaError_t Softmaxf32(float* loss, float* output, float* outputGrad, int* target, int outputSize, int batchSize);
cudaError_t SoftmaxPredictf32(float* input, int* target, int inputSize, int batchSize);
