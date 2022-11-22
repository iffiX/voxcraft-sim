#ifndef VX3_H
#define VX3_H

#include <string>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#define COLORCODE_RED "\033[0;31m" 
#define COLORCODE_BOLD_RED "\033[1;31m\n" 
#define COLORCODE_GREEN "\033[0;32m" 
#define COLORCODE_BLUE "\033[0;34m" 
#define COLORCODE_RESET "\033[0m" 

//#define DEBUG_LINE printf("%s(%d): %s\n", __FILE__, __LINE__, u_format_now("at %M:%S").c_str());
#define CUDA_DEBUG_LINE(str) {printf("%s(%d): %s\n", __FILE__, __LINE__, str);}
#define DEBUG_LINE_

#ifndef CUDA_ERROR_CHECK
    __device__ __host__ inline void CUDA_ERROR_CHECK_OUTPUT(cudaError_t code, const char *file, int line, bool abort=false) {
        if (code != cudaSuccess) {
            printf(COLORCODE_BOLD_RED "%s(%d): CUDA Function Error: %s \n" COLORCODE_RESET, file, line, cudaGetErrorString(code));
            if (abort) assert(0);
        }
    }
    #define CUDA_ERROR_CHECK(ans) { CUDA_ERROR_CHECK_OUTPUT((ans), __FILE__, __LINE__); }
#endif

// Verbose calls to cuda runtime with error checking
#define VcudaMemGetInfo(a,b) {CUDA_ERROR_CHECK(cudaMemGetInfo(a,b))}
#define VcudaDeviceSetLimit(a,b) {CUDA_ERROR_CHECK(cudaDeviceSetLimit(a,b))}
#define VcudaSetDevice(a) {CUDA_ERROR_CHECK(cudaSetDevice(a))}
#define VcudaGetDeviceCount(a) {CUDA_ERROR_CHECK(cudaGetDeviceCount(a))}
#define VcudaMemcpy(a,b,c,d)  {CUDA_ERROR_CHECK(cudaMemcpy(a,b,c,d))}
#define VcudaMemcpyAsync(a,b,c,d,e)  {CUDA_ERROR_CHECK(cudaMemcpyAsync(a,b,c,d,e))}
#define VcudaMalloc(a,b) {CUDA_ERROR_CHECK(cudaMalloc(a,b))}
#define VcudaFree(a) {CUDA_ERROR_CHECK(cudaFree(a))}
#define VcudaGetLastError() {CUDA_ERROR_CHECK(cudaGetLastError())}
#define VcudaDeviceSynchronize() {CUDA_ERROR_CHECK(cudaDeviceSynchronize())}
#define VcudaMemcpyHostToDevice cudaMemcpyHostToDevice
#define VcudaMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define CUDA_CHECK_AFTER_CALL() {CUDA_ERROR_CHECK(cudaGetLastError());}

#endif // VX3_H
