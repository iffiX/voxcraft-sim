#ifndef VX3_SEARCH_CUH
#define VX3_SEARCH_CUH

#include "utils/vx3_def.h"

struct GroupThread {
    Vindex gid;
    Vindex tid;
};

/**
 * @param tid
 * @param group_sizes_prefix_sum sum of group sizes,
 * the first element is the size of the first group,
 * the last element is the size of all groups
 * Eg:
 * group size = [1, 2, 3, 4, 5],
 * prefix sum = [1, 3, 6, 10, 15]
 * @param group_num
 * @return
 */
__device__ GroupThread binary_group_search(Vindex tid, const Vsize *group_sizes_prefix_sum, Vsize group_num)
{
    if (group_sizes_prefix_sum[group_num - 1] <= tid) {
        return {NULL_INDEX, NULL_INDEX};
    }
    else if (group_num == 1){
        return {0, tid};
    }
    else {
        // At least 2 groups, and tid is within max boundary
        Vindex i_min = 0, i_max = group_num - 1;
        while (i_max > i_min) {
            Vindex i_mid = (i_min + i_max) / 2;
            Vsize low = group_sizes_prefix_sum[i_mid];
            bool has_low = low <= tid;
            bool has_high = group_sizes_prefix_sum[i_mid + 1] > tid;
            if (has_low && has_high) {
                return {i_mid + 1, tid - low};
            } else if (has_low) {
                i_min = i_mid;
            } else {
                i_max = i_mid;
                if (i_max == 0)
                    return {0, tid};
            }
        }
    }
}

#endif // VX3_SEARCH_CUH