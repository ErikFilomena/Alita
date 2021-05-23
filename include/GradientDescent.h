#pragma once

#include "CommonOp.cuh"
#include "AlitaCore.h"

#include <vector>
#include "Layer.h"

#define LayersVector std::vector<Layerf32*>
#include <vector>
#include "Layer.h"

#define LayersVector std::vector<Layerf32*>


cudaError_t GradientDescentf32(LayersVector layers, float alpha, float lambda);