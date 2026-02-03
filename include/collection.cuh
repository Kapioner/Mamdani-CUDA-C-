#pragma once
#include <cuda_runtime.h>
#include <cmath>

//Funkcja trójkątna
__host__ __device__ float trimf(float x, float a, float b, float c) {
    return fmaxf(0.0f, fminf((x - a) / (b - a), (c - x) / (c - b)));
}
//Funkcja trapezowa lewostronna
__host__ __device__ float trapmf_left(float x, float b, float c) {
    return fmaxf(0.0f, fminf(1.0f, (c - x) / (c - b)));
}
//Funkcja trapezowa prawostronna
__host__ __device__ float trapmf_right(float x, float a, float b) {
    return fmaxf(0.0f, fminf((x - a) / (b - a), 1.0f));
}