#pragma once
#include "AlitaCore.h"
#include "CommonOp.cuh"
#include "Layer.h"
#include "Conv2dLayer.h"


#define LayersVector std::vector<Layerf32*>
#define ConvLayersVector std::vector<Conv2dLayerf32*>


cudaError_t GradientDescentf32(LayersVector layers, float alpha, float lambda);
cudaError_t GradientDescentf32(ConvLayersVector layers, float alpha, float lambda);