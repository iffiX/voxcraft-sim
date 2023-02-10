#ifndef VX3_CONF_H
#define VX3_CONF_H
// #define DEBUG_CUDA_KERNEL  true

#define VX3_VOXELYZE_KERNEL_ALLOCATE_FRAME_NUM 500

// 65536 is the 64KB constant memory, we reserve another 1KB
// for passing other arguments, and subtract a kernel size
// for additional padding
#define VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE        (((65536 - 1024) / sizeof(VX3_VoxelyzeKernel)) - 1)

#define VX3_MATH_TREE_MAX_STACK_SIZE              4
#define VX3_MATH_TREE_MAX_EXPRESSION_TOKENS       16


#endif