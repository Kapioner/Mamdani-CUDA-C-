#include <vector>
#include <iostream>
#include "simulation.cuh"

const int N = 12341623; // Ilość próbek

int main() {
    // Przygotowanie wejść i wyjść systemu
    std::vector<float> data_temp(N);
    std::vector<float> data_hum(N);
    std::vector<float> results_cpu(N);
    std::vector<float> results_gpu(N);

    // Generacja danych
    generateData(data_temp, data_hum);

    // Uruchomienie symulacji CPU z pomiarem czasu
    double time_cpu = runCpuSimulation(data_temp, data_hum, results_cpu);
    std::cout << "Czas CPU: " << time_cpu << " ms" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    
    // Uruchomienie symulacji GPU z pomiarem czasu
    double time_gpu = runGpuSimulation(data_temp, data_hum, results_gpu);
    std::cout << "Czas GPU: " << time_gpu << " ms" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Roznica: " << time_cpu / time_gpu << "x" << std::endl;

    return 0;
}