#pragma once
#include <vector>

void generateData(std::vector<float>& temp, std::vector<float>& hum);

double runCpuSimulation(const std::vector<float>& temp, const std::vector<float>& hum, std::vector<float>& out);

double runGpuSimulation(const std::vector<float>& h_temp, const std::vector<float>& h_hum, std::vector<float>& h_out);