// Adapted from https://github.com/mark-poscablo/gpu-sum-reduction

#ifndef VX3_REDUCE_CUH
#define VX3_REDUCE_CUH
#include "utils/vx3_cuda.cuh"
#include "utils/vx3_def.h"
#define MAX_BLOCK_SZ 1024

template <typename T, typename ReduceOp>
__global__ void block_reduce(const T *d_in, T *d_out, Vsize d_in_len, T init_value) {
    auto op = ReduceOp();
    extern __shared__ unsigned char s_mem[];
    T *s_out = reinterpret_cast<T *>(s_mem);

    Vindex elem_offset = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    Vindex local_tid = threadIdx.x;

    // Clear shared memory
    // Especially important when padding shmem for
    //  non-power of 2 sized input
    s_out[threadIdx.x] = init_value;
    s_out[threadIdx.x + blockDim.x] = init_value;

    __syncthreads();

    // Copy d_in to shared memory per block
    if (elem_offset < d_in_len) {
        s_out[threadIdx.x] = d_in[elem_offset];
        if (elem_offset + blockDim.x < d_in_len)
            s_out[threadIdx.x + blockDim.x] = d_in[elem_offset + blockDim.x];
    }
    __syncthreads();

    // Actually do the reduction
    for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
        if (local_tid < s) {
            s_out[local_tid] = op(s_out[local_tid], s_out[local_tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (local_tid == 0) {
        d_out[blockIdx.x] = s_out[0];
    }
}

Vsize getReduceBufferSize(Vsize elem_size, Vsize elem_num) {
    // Set up number of threads and blocks
    // If input size is not power of two, the remainder will still need a whole block
    // Thus, number of blocks must be the least number of 2048-blocks greater than the
    // input size
    Vsize block_size = MAX_BLOCK_SZ; // Halve the block size due to reduce3() and further

    // due to binary tree nature of algorithm
    Vsize max_elems_per_block = block_size * 2;

    Vsize grid_size = CEIL(elem_num, max_elems_per_block);
    return elem_size * grid_size;
}

/**
 * @tparam T
 * @tparam ReduceOp
 * @tparam InitValue Value used to initialize the shared memory
 * @param host_buffer Pinned memory allocated by cudaMallocHost. size >= sizeof(T)
 * @param d_in Input device memory.
 * @param d_reduce_buffer Reduce buffer of size >= getReduceBufferSize(sizeof(T), element
 * number)
 * @param d_in_len Input element number.
 * @param stream
 * Note: since required reduce buffer size is always smaller than the input size,
 *  you can use 2 same size reduce buffers the same size as the input (or just set
 * d_reduce_buffer2 = d_in)
 */
template <typename T, typename ReduceOp>
T reduce(void *host_buffer, const void *d_in, void *d_reduce_buffer1,
         void *d_reduce_buffer2, Vsize d_in_len, const cudaStream_t &stream,
         T init_value = 0) {
    ((T *)host_buffer)[0] = init_value;

    Vsize block_size = MAX_BLOCK_SZ;
    Vsize max_elems_per_block = block_size * 2;
    Vsize grid_size = CEIL(d_in_len, max_elems_per_block);

    // Perform block level reduce
    block_reduce<T, ReduceOp>
        <<<grid_size, block_size, sizeof(T) * max_elems_per_block, stream>>>(
            (const T *)d_in, (T *)d_reduce_buffer1, d_in_len, init_value);

    if (d_in_len < max_elems_per_block) {
        // The result is ready in d_reduce_buffer since there is only 1 block
        VcudaStreamSynchronize(stream);
        CUDA_CHECK_AFTER_CALL();
        VcudaMemcpyAsync(host_buffer, d_reduce_buffer1, sizeof(T), cudaMemcpyDeviceToHost,
                         stream);
        VcudaStreamSynchronize(stream);
        return ((T *)host_buffer)[0];
    } else
        // Reduce recursively
        return reduce<T, ReduceOp>(host_buffer, d_reduce_buffer1, d_reduce_buffer2,
                                   d_reduce_buffer1, grid_size, stream, init_value);
}

template <typename T> struct maxReduce {
    __device__ T operator()(const T &a, const T &b) { return MAX(a, b); }
};

template <typename T> struct sumReduce {
    __device__ T operator()(const T &a, const T &b) { return a + b; }
};

template <typename T> struct orReduce {
    __device__ T operator()(const T &a, const T &b) { return a || b; }
};
#endif // VX3_REDUCE_CUH
