#include "collection.cuh"
#include "mamdani_fuzzy_system.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

const int DEFUZZ_STEPS = 2;

__host__ __device__ float computeMamdani(float temp, float hum) {
    // Ustalenie zbiorów rozmytych
    float t_cold   = trapmf_left(temp, 10.0f, 20.0f);
    float t_medium = trimf(temp, 15.0f, 25.0f, 35.0f);
    float t_hot    = trapmf_right(temp, 30.0f, 40.0f);
    float h_low    = trapmf_left(hum, 30.0f, 60.0f);
    float h_high   = trapmf_right(hum, 40.0f, 70.0f);

    // Tablica reguł
    float r1 = t_cold;
    float r2 = fminf(t_medium, h_low);
    float r3 = fmaxf(t_hot, h_high);

    //Grupowanie reguł
    float slow_act = fmaxf(r1, r2);
    float fast_act = r3;
    float num = 0.0f;
    float den = 0.0f;
    
    //Obliczenie wyjść dla danego kroku defuzyfikacji
    for (int i = 0; i <= DEFUZZ_STEPS; ++i) {
        float x = i * (100.0f / DEFUZZ_STEPS);
        
        float mu_slow = fminf(trapmf_left(x, 30.0f, 60.0f), slow_act);
        float mu_fast = fminf(trapmf_right(x, 40.0f, 70.0f), fast_act);
        
        float mu_final = fmaxf(mu_slow, mu_fast);

        num += x * mu_final;
        den += mu_final;
    }

    return (den == 0.0f) ? 0.0f : num / den;
}

// Implementacja funkcji CUDA obsługującego obliczanie wartości systemu Mamdaniego na wielu wątkach
__global__ void fuzzyKernel(const float* d_temp, const float* d_hum, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = computeMamdani(d_temp[idx], d_hum[idx]);
    }
}


