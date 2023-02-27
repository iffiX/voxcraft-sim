// Adapted from https://github.com/mark-poscablo/gpu-sum-reduction

#ifndef VX3_REDUCE_CUH
#define VX3_REDUCE_CUH

#include "utils/vx3_cuda.cuh"
#include "utils/vx3_def.h"
#include "utils/vx3_search.cuh"
#include <vector>

#define MAX_GROUP_REDUCE_LEVEL 5

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

/**
 * @tparam T
 * @param d_in   "Packed" input array
 * @param d_out  Array with input groups aligned to multiples of 2 * CUDA_MAX_BLOCKSIZE
 * @param d_in_sizes Input group sizes
 * @param d_in_sizes_psum Input group offsets (i.e. prefix sum of input group sizes,
 * starting with 0)
 * @param d_out_sizes_psum Prefix sum of output group sizes,
 * note it starts with the size of the first output group
 * @param group_num Number of groups
 * @param init_value Initial value for padding
 */
template <typename T>
__global__ void block_align(const T *d_in, T *d_out, const Vsize *d_in_sizes,
                            const Vsize *d_in_offsets, const Vsize *d_out_sizes_psum,
                            Vsize group_num, T init_value) {

    Vindex tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, d_out_sizes_psum, group_num);
    if (gt.gid == NULL_INDEX)
        return;

    if (gt.tid < d_in_sizes[gt.gid]) {
        d_out[tid] = d_in[gt.tid + d_in_offsets[gt.gid]];
    } else {
        d_out[tid] = init_value;
    }
}

Vsize getReduceBufferSize(Vsize elem_size, Vsize elem_num) {
    // Set up number of threads and blocks
    // If input size is not power of two, the remainder will still need a whole block
    // Thus, number of blocks must be the least number of 2048-blocks greater than the
    // input size

    // due to binary tree nature of algorithm
    Vsize max_elems_per_block = CUDA_MAX_BLOCK_SIZE * 2;

    Vsize grid_size = CEIL(elem_num, max_elems_per_block);
    return elem_size * grid_size;
}

Vsize getReduceByGroupBufferSize(Vsize elem_size, std::vector<Vsize> group_elem_num) {
    Vsize total = 0;

    // due to binary tree nature of algorithm
    Vsize max_elems_per_block = CUDA_MAX_BLOCK_SIZE * 2;

    for (auto elem_num : group_elem_num)
        total += ALIGN(elem_num, max_elems_per_block);
    return elem_size * total;
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
 * Note: reduce implicitly synchronizes stream before returning results
 */
template <typename T, typename ReduceOp>
T reduce(void *host_buffer, const void *d_in, void *d_reduce_buffer1,
         void *d_reduce_buffer2, Vsize d_in_len, const cudaStream_t &stream,
         T init_value = 0) {
    ((T *)host_buffer)[0] = init_value;

    Vsize block_size = CUDA_MAX_BLOCK_SIZE;
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

template <typename T, typename ReduceOp>
std::vector<T>
_reduce_by_group(void *host_buffer, const void *d_in, void *d_reduce_buffer1,
                 void *d_reduce_buffer2, Vsize *h_sizes, Vsize *d_sizes, Vsize group_num,
                 Vsize level, Vsize level_num, const cudaStream_t &stream, T init_value) {
    Vsize block_size, grid_size;

    block_size = CUDA_MAX_BLOCK_SIZE;
    // Perform align
    // The number of total needed threads is equal to
    // the last prefix sum of the output group sizes
    grid_size = CEIL(h_sizes[group_num * 3 - 1], block_size);
    block_align<T><<<grid_size, block_size, 0, stream>>>(
        (const T *)d_in, (T *)d_reduce_buffer1, d_sizes, d_sizes + group_num,
        d_sizes + group_num * 2, group_num, init_value);

    // Perform block level reduce
    // d_in_len argument is also equal to the last
    // prefix sum of the output group sizes
    Vsize max_elems_per_block = block_size * 2;
    Vsize d_in_len = h_sizes[group_num * 3 - 1];
    grid_size = CEIL(d_in_len, max_elems_per_block);
    block_reduce<T, ReduceOp>
        <<<grid_size, block_size, sizeof(T) * max_elems_per_block, stream>>>(
            (const T *)d_reduce_buffer1, (T *)d_reduce_buffer2, d_in_len, init_value);

    if (level + 1 == level_num) {
        // The result is ready in d_reduce_buffer2
        VcudaMemcpyAsync(host_buffer, d_reduce_buffer2, sizeof(T) * group_num,
                         cudaMemcpyDeviceToHost, stream);
        VcudaStreamSynchronize(stream);
        std::vector<T> result;
        for (Vsize i = 0; i < group_num; i++)
            result.emplace_back(((T *)host_buffer)[i]);
        return std::move(result);
    } else
        // Reduce recursively
        return _reduce_by_group<T, ReduceOp>(
            host_buffer, d_reduce_buffer2, d_reduce_buffer1, d_reduce_buffer2,
            h_sizes + group_num * 3, d_sizes + group_num * 3, group_num, level + 1,
            level_num, stream, init_value);
}

/**
 *
 * @tparam T
 * @tparam ReduceOp
 * @param host_buffer Pinned memory allocated by cudaMallocHost. size >= sizeof(T) *
 * number of groups
 * @param d_in
 * @param d_reduce_buffer1
 * @param d_reduce_buffer2
 * @param h_sizes_buffer sizeof(Vsize) * group_num * 3 * MAX_GROUP_REDUCE_LEVEL
 * ( 5 levels are more than enough, which is 2^55 elements)
 * @param d_sizes_buffer sizeof(Vsize) * group_num * 3 * MAX_GROUP_REDUCE_LEVEL
 * @param d_in_group_len
 * @param stream
 * @param init_value
 * Note: To just use 2 reduce buffers, set d_reduce_buffer2 = d_in)
 * Note: reduce_by_group implicitly synchronizes stream before returning results
 */
template <typename T, typename ReduceOp>
std::vector<T> reduce_by_group(void *host_buffer, const void *d_in,
                               void *d_reduce_buffer1, void *d_reduce_buffer2,
                               Vsize *h_sizes_buffer, Vsize *d_sizes_buffer,
                               const std::vector<Vsize> &d_in_group_len,
                               const cudaStream_t &stream, T init_value = 0) {
    for (size_t i = 0; i < d_in_group_len.size(); i++)
        ((T *)host_buffer)[i] = init_value;

    // Compute sizes
    auto group_sizes = d_in_group_len;
    Vsize max_elems_per_block = CUDA_MAX_BLOCK_SIZE * 2;
    auto *h_sizes = h_sizes_buffer;
    Vsize level_num = 1;
    bool is_all_reduced;
    for (; level_num <= MAX_GROUP_REDUCE_LEVEL; level_num++) {
        // Store d_in_sizes
        for (auto group_size : group_sizes) {
            *h_sizes = group_size;
            h_sizes++;
        }

        // Compute input group offsets
        Vsize offset = 0;
        for (auto group_size : group_sizes) {
            *h_sizes = offset;
            offset += group_size;
            h_sizes++;
        }

        // Compute output group sizes and prefix sums
        Vsize psum = 0;
        for (auto group_size : group_sizes) {
            psum += ALIGN(group_size, max_elems_per_block);
            *h_sizes = psum;
            h_sizes++;
        }
        // Update group sizes
        is_all_reduced = true;
        for (auto &group_size : group_sizes) {
            group_size = CEIL(group_size, max_elems_per_block);
            if (group_size > 1)
                is_all_reduced = false;
        }
        if (is_all_reduced)
            break;
    }
    if (not is_all_reduced)
        throw std::invalid_argument("Not all elements are reduced");

    VcudaMemcpyAsync(d_sizes_buffer, h_sizes_buffer,
                     sizeof(Vsize) * d_in_group_len.size() * 3 * MAX_GROUP_REDUCE_LEVEL,
                     cudaMemcpyHostToDevice, stream);
    return std::move(_reduce_by_group<T, ReduceOp>(
        host_buffer, d_in, d_reduce_buffer1, d_reduce_buffer2, h_sizes_buffer,
        d_sizes_buffer, d_in_group_len.size(), 0, level_num, stream, init_value));
}

template <typename T> struct MaxReduce {
    __device__ T operator()(const T &a, const T &b) { return MAX(a, b); }
};

template <typename T> struct SumReduce {
    __device__ T operator()(const T &a, const T &b) { return a + b; }
};

template <typename T> struct OrReduce {
    __device__ T operator()(const T &a, const T &b) { return a || b; }
};

#endif // VX3_REDUCE_CUH
