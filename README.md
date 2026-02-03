To compile the project type:
```
nvcc .\src\main.cu .\src\simulation.cu .\src\mamdani_fuzzy_system.cu -I./include -o .\run\mamdani
```

Compute times for given samples and diffusion steps:

N = 10000000, DEFUZZ_STEPS = 2
```
CPU: 1729.52 ms ms, GPU 6.4701 ms, diff: 267.309x
```
N = 50000, DEFUZZ_STEPS = 2
```
CPU: 10.7726 ms, GPU: 7.8368 ms, diff: 1.37462x
```
N = 50000, DEFUZZ_STEPS = 100
```
CPU: 178.984 ms, GPU: 5.556 ms, diff: 32.2146x
```

This project is licensed under the MIT License - see the LICENSE.md file for details
