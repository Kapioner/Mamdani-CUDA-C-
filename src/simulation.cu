#include <chrono>
#include <random>
#include <iostream>
#include "mamdani_fuzzy_system.cuh"
#include "simulation.cuh"

const int N = 50000; // Ilość próbek

// Generowanie danych testowych
void generateData(std::vector<float>& temp, std::vector<float>& hum) {
    std::cout << "Generowanie " << N << " losowych probek" << std::endl;
    std::srand(time(0));
    for (int i = 0; i < N; i++) {
        temp[i] = static_cast<float>(rand() % 51);
        hum[i] = static_cast<float>(rand() % 101);
    }
}

// Uruchomienie symulacji systemu Mamdaniego na CPU i pomiar czasu poprzez chrono
double runCpuSimulation(const std::vector<float>& temp, const std::vector<float>& hum, std::vector<float>& out) {
    std::cout << "CPU: Start obliczen" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        out[i] = computeMamdani(temp[i], hum[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Uruchomienie symulacji systemu Mamdaniego na GPU i pomiar czasu poprzez chrono
double runGpuSimulation(const std::vector<float>& h_temp, const std::vector<float>& h_hum, std::vector<float>& h_out) {
    std::cout << "GPU: start obliczen" << std::endl;
    float *d_temp, *d_hum, *d_out;
    size_t bytes = N * sizeof(float);
    //Początek pomiaru
    auto start = std::chrono::high_resolution_clock::now();

    //Alokacja pamięci
    cudaMalloc(&d_temp, bytes);
    cudaMalloc(&d_hum, bytes);
    cudaMalloc(&d_out, bytes);

    //Kopiowanie z pamięci hosta(CPU) do device(GPU)
    cudaMemcpy(d_temp, h_temp.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hum, h_hum.data(), bytes, cudaMemcpyHostToDevice);

    //Uruchomienie kernela i synchronizacja
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    fuzzyKernel<<<gridSize, blockSize>>>(d_temp, d_hum, d_out, N);
    cudaDeviceSynchronize();

    //Kopipwanie danych do hosta
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    //Pomiar czasu koncowego
    auto end = std::chrono::high_resolution_clock::now();
    
    //Zwolnienie pamieci
    cudaFree(d_temp);
    cudaFree(d_hum);
    cudaFree(d_out);

    return std::chrono::duration<double, std::milli>(end - start).count();
}