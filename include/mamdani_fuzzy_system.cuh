#pragma once
#include <cuda_runtime.h>

__host__ __device__ float computeMamdani(float temp, float hum);
__global__ void fuzzyKernel(const float* d_temp, const float* d_hum, float* d_out, int n);