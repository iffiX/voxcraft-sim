#ifndef VX3_H
#define VX3_H

#include <assert.h>
#include <chrono>
#include <ctime>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#define COLORCODE_RED "\033[0;31m"
#define COLORCODE_BOLD_RED "\033[1;31m\n"
#define COLORCODE_GREEN "\033[0;32m"
#define COLORCODE_BLUE "\033[0;34m"
#define COLORCODE_RESET "\033[0m"

//#define DEBUG_LINE printf("%s(%d): %s\n", __FILE__, __LINE__, u_format_now("at
//%M:%S").c_str());
#define CUDA_DEBUG_LINE(str)                                                             \
    { printf("%s(%d): %s\n", __FILE__, __LINE__, str); }
#define DEBUG_LINE_

#ifndef CUDA_ERROR_CHECK
__device__ __host__ inline void CUDA_ERROR_CHECK_OUTPUT(cudaError_t code,
                                                        const char *file, int line,
                                                        bool abort = false) {
    if (code != cudaSuccess) {
        printf(COLORCODE_BOLD_RED "%s(%d): CUDA Function Error: %s \n" COLORCODE_RESET,
               file, line, cudaGetErrorString(code));
        if (abort)
            assert(0);
    }
}
#define CUDA_ERROR_CHECK(ans)                                                            \
    { CUDA_ERROR_CHECK_OUTPUT((ans), __FILE__, __LINE__); }
#endif

// Verbose calls to cuda runtime with error checking
#define VcudaMemGetInfo(free, total)                                                     \
    { CUDA_ERROR_CHECK(cudaMemGetInfo(free, total)) }
#define VcudaDeviceSetLimit(property, value)                                             \
    { CUDA_ERROR_CHECK(cudaDeviceSetLimit(property, value)) }
#define VcudaSetDevice(device)                                                           \
    { CUDA_ERROR_CHECK(cudaSetDevice(device)) }
#define VcudaGetDeviceCount(count)                                                       \
    { CUDA_ERROR_CHECK(cudaGetDeviceCount(count)) }
#define VcudaStreamCreate(stream)                                                        \
    { CUDA_ERROR_CHECK(cudaStreamCreate(stream)) }
#define VcudaStreamDestroy(stream)                                                       \
    { CUDA_ERROR_CHECK(cudaStreamDestroy(stream)) }
#define VcudaMemcpyAsync(dst, src, count, kind, stream)                                  \
    { CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, src, count, kind, stream)) }
#define VcudaMallocAsync(dst, size, stream)                                              \
    { CUDA_ERROR_CHECK(cudaMallocAsync(dst, size, stream)) }
#define VcudaFreeAsync(mem, stream)                                                      \
    { CUDA_ERROR_CHECK(cudaFreeAsync(mem, stream)) }
#define VcudaMallocHost(dst, size)                                                       \
    { CUDA_ERROR_CHECK(cudaMallocHost(dst, size)) }
#define VcudaFreeHost(mem)                                                               \
    { CUDA_ERROR_CHECK(cudaFreeHost(mem)) }
#define VcudaGetLastError()                                                              \
    { CUDA_ERROR_CHECK(cudaGetLastError()) }
#define VcudaStreamSynchronize(stream)                                                   \
    { CUDA_ERROR_CHECK(cudaStreamSynchronize(stream)) }
#define CUDA_CHECK_AFTER_CALL()                                                          \
    { CUDA_ERROR_CHECK(cudaGetLastError()); }

#endif // VX3_H
