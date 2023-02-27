#include "utils/vx3_conf.h"
#include "utils/vx3_reduce.cuh"
#include "vx3/vx3_simulation_record.h"
#include "vx3_voxelyze_kernel.cuh"
#include <fmt/format.h>
#include <iostream>

using namespace std;
using namespace fmt;

struct GroupSizesPrefixSum {
    Vsize sums[VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE] = {0};
};

struct GroupToKernelIndex {
    Vindex index[VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE] = {0};
};

/* Tools */
template <typename T> inline pair<int, int> getGridAndBlockSize(T func, size_t item_num) {
    int min_grid_size, grid_size, block_size;

    if (item_num < 256) {
        // For small simulations, fit them into 1 block
        // No more than 1024 (cuda max block thread num)
        block_size = 256;
    } else {
        // Dynamically calculate blockSize
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func, 0,
                                           item_num);
    }
    grid_size = CEIL(item_num, block_size);
    block_size = MIN(item_num, block_size);
    return make_pair(grid_size, block_size);
}

void computePrefixSum(GroupSizesPrefixSum &sizes, Vsize group_num) {
    for (Vsize i = 1; i < group_num; i++) {
        sizes.sums[i] = sizes.sums[i] + sizes.sums[i - 1];
    }
}

template <typename FuncType, typename... Args>
void runFunction(FuncType func, const VX3_VoxelyzeKernelBatchExecutor &exec, bool is_link,
                 int block_size, const vector<size_t> &kernel_indices, Args &&...args) {
    if (kernel_indices.empty())
        return;

    vector<Vsize> group_len;
    GroupSizesPrefixSum p_sum;
    GroupToKernelIndex k_index;
    vector<Vsize> non_empty_kernel_relative_indices;
    Vsize relative_idx = 0;
    for (auto idx : kernel_indices) {
        size_t link_num = exec.kernels[idx]->ctx.links.size();
        size_t voxel_num = exec.kernels[idx]->ctx.voxels.size();
        if (is_link and link_num > 0) {
            size_t size = non_empty_kernel_relative_indices.size();
            p_sum.sums[size] = link_num;
            group_len.push_back(link_num);
            k_index.index[size] = idx;
            non_empty_kernel_relative_indices.push_back(relative_idx);
        } else if (not is_link and voxel_num > 0) {
            size_t size = non_empty_kernel_relative_indices.size();
            p_sum.sums[size] = voxel_num;
            group_len.push_back(voxel_num);
            k_index.index[size] = idx;
            non_empty_kernel_relative_indices.push_back(relative_idx);
        }
        relative_idx++;
    }
    Vsize group_num = non_empty_kernel_relative_indices.size();

    computePrefixSum(p_sum, group_num);
    Vsize total_threads = p_sum.sums[non_empty_kernel_relative_indices.size() - 1];

    pair<int, int> size;
    if (block_size > 0) {
        size.first = CEIL(total_threads, block_size);
        size.second = block_size;
    } else
        size = getGridAndBlockSize(func, total_threads);
    func<<<size.first, size.second, 0, exec.stream>>>(
        exec.d_kernels, p_sum, k_index, group_num, std::forward<Args>(args)...);
    CUDA_CHECK_AFTER_CALL();
}

/**
 * Note: runFunctionAndReduce implicitly synchronizes stream before returning results
 * due to the call to reduce_by_group
 */
template <typename ReduceOp, typename FuncType, typename ResultType>
void runFunctionAndReduce(FuncType func, vector<ResultType> &result,
                          ResultType init_value,
                          const VX3_VoxelyzeKernelBatchExecutor &exec, bool is_link,
                          const vector<size_t> &kernel_indices) {
    if (kernel_indices.empty())
        return;

    if (result.size() != kernel_indices.size())
        throw std::invalid_argument(
            "Result vector size must be the same as kernel indices size.");

    vector<Vsize> group_len;
    GroupSizesPrefixSum p_sum;
    GroupToKernelIndex k_index;
    vector<Vsize> non_empty_kernel_relative_indices;
    Vsize relative_idx = 0;
    for (auto idx : kernel_indices) {
        size_t link_num = exec.kernels[idx]->ctx.links.size();
        size_t voxel_num = exec.kernels[idx]->ctx.voxels.size();
        if (is_link and link_num > 0) {
            size_t size = non_empty_kernel_relative_indices.size();
            p_sum.sums[size] = link_num;
            group_len.push_back(link_num);
            k_index.index[size] = idx;
            non_empty_kernel_relative_indices.push_back(relative_idx);
        } else if (not is_link and voxel_num > 0) {
            size_t size = non_empty_kernel_relative_indices.size();
            p_sum.sums[size] = voxel_num;
            group_len.push_back(voxel_num);
            k_index.index[size] = idx;
            non_empty_kernel_relative_indices.push_back(relative_idx);
        }
        relative_idx++;
    }
    Vsize group_num = non_empty_kernel_relative_indices.size();

    computePrefixSum(p_sum, group_num);
    Vsize total_threads = p_sum.sums[non_empty_kernel_relative_indices.size() - 1];

    auto size = getGridAndBlockSize(func, total_threads);
    func<<<size.first, size.second, 0, exec.stream>>>(
        exec.d_kernels, p_sum, k_index, group_num, (ResultType *)exec.d_reduce1);
    CUDA_CHECK_AFTER_CALL();

    //    // For debugging`
    //    VcudaStreamSynchronize(exec.stream);
    //    ResultType *tmp;
    //    VcudaMallocHost(&tmp, sizeof(ResultType) *
    //    p_sum.sums[non_empty_kernel_relative_indices.size() - 1]); VcudaMemcpyAsync(tmp,
    //    exec.d_reduce1,
    //                     sizeof(ResultType) *
    //                     p_sum.sums[non_empty_kernel_relative_indices.size() - 1],
    //                     cudaMemcpyDeviceToHost, exec.stream);
    //    VcudaStreamSynchronize(exec.stream);
    //    size_t offset = 0;
    //    for (size_t k = 0; k < non_empty_kernel_relative_indices.size(); k++) {
    //        std::cout << "Kernel " << k << ": ";
    //        for(size_t i = offset; i < p_sum.sums[k]; i++) {
    //            std::cout << ((float*)tmp)[i] << " ";
    //        }
    //        std::cout << std::endl;
    //        offset = p_sum.sums[k];
    //    }

    auto partial_result = reduce_by_group<ResultType, ReduceOp>(
        exec.h_reduce_output, exec.d_reduce1, exec.d_reduce2, exec.d_reduce1,
        exec.h_sizes, exec.d_sizes, group_len, exec.stream, init_value);
    //
    //    VcudaFreeHost(tmp);

    CUDA_CHECK_AFTER_CALL();
    for (size_t i = 0; i < non_empty_kernel_relative_indices.size(); i++) {
        result[non_empty_kernel_relative_indices[i]] = partial_result[i];
    }
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::isStopConditionMet
 *****************************************************************************/
bool VX3_VoxelyzeKernel::isStopConditionMet() const {
    return VX3_MathTree::eval(current_center_of_mass.x, current_center_of_mass.y,
                              current_center_of_mass.z, (Vfloat)collision_count, time,
                              recent_angle, target_closeness, num_close_pairs,
                              (int)ctx.voxels.size(), stop_condition_formula) > 0;
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::computeFitness
 *****************************************************************************/
Vfloat VX3_VoxelyzeKernel::computeFitness() const {
    Vec3f offset = current_center_of_mass - initial_center_of_mass;
    return VX3_MathTree::eval(offset.x, offset.y, offset.z, (Vfloat)collision_count, time,
                              recent_angle, target_closeness, num_close_pairs,
                              (int)ctx.voxels.size(), fitness_function);
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::adjustRecordFrameStorage
 *****************************************************************************/
void VX3_VoxelyzeKernel::adjustRecordFrameStorage(size_t required_size,
                                                  cudaStream_t stream) {
    if (frame_storage_size < required_size) {
        unsigned long *new_d_steps;
        Vfloat *new_d_time_points;
        VX3_SimulationLinkRecord *new_d_link_record;
        VX3_SimulationVoxelRecord *new_d_voxel_record;
        size_t new_frame_capacity =
            frame_storage_size + VX3_VOXELYZE_KERNEL_ALLOCATE_FRAME_NUM;
        VcudaStreamSynchronize(stream);
        VcudaMallocAsync(&new_d_steps, sizeof(unsigned long) * new_frame_capacity,
                         stream);
        VcudaMallocAsync(&new_d_time_points, sizeof(Vfloat) * new_frame_capacity, stream);
        VcudaMallocAsync(&new_d_link_record,
                         sizeof(VX3_SimulationLinkRecord) * ctx.links.size() *
                             new_frame_capacity,
                         stream);
        VcudaMallocAsync(&new_d_voxel_record,
                         sizeof(VX3_SimulationVoxelRecord) * ctx.voxels.size() *
                             new_frame_capacity,
                         stream);

        if (frame_storage_size > 0) {
            VcudaMemcpyAsync(new_d_steps, d_steps,
                             sizeof(unsigned long) * frame_storage_size,
                             cudaMemcpyDeviceToDevice, stream);
            VcudaMemcpyAsync(new_d_time_points, d_time_points,
                             sizeof(Vfloat) * frame_storage_size,
                             cudaMemcpyDeviceToDevice, stream);
            VcudaMemcpyAsync(new_d_link_record, d_link_record,
                             sizeof(VX3_SimulationLinkRecord) * ctx.links.size() *
                                 frame_storage_size,
                             cudaMemcpyDeviceToDevice, stream);
            VcudaMemcpyAsync(new_d_voxel_record, d_voxel_record,
                             sizeof(VX3_SimulationVoxelRecord) * ctx.voxels.size() *
                                 frame_storage_size,
                             cudaMemcpyDeviceToDevice, stream);
        }
        VcudaFreeAsync(d_steps, stream);
        VcudaFreeAsync(d_time_points, stream);
        VcudaFreeAsync(d_link_record, stream);
        VcudaFreeAsync(d_voxel_record, stream);
        VcudaStreamSynchronize(stream);
        d_steps = new_d_steps;
        d_time_points = new_d_time_points;
        d_link_record = new_d_link_record;
        d_voxel_record = new_d_voxel_record;
        frame_storage_size = new_frame_capacity;
    }
}

/*****************************************************************************
 * VX3_VoxelyzeKernel device sub-routines
 *****************************************************************************/
__device__ void VX3_VoxelyzeKernel::updateLinks(Vindex local_id) {
    if (local_id < ctx.links.size()) {
        Vindex voxel_neg = L_G(local_id, voxel_neg);
        Vindex voxel_pos = L_G(local_id, voxel_pos);
        Vindex voxel_neg_mat = V_G(voxel_neg, voxel_material);
        Vindex voxel_pos_mat = V_G(voxel_pos, voxel_material);

        if (L_G(local_id, removed))
            return;
        if (VM_G(voxel_neg_mat, fixed) && VM_G(voxel_pos_mat, fixed))
            return;
        if (L_G(local_id, is_detached))
            return;
        VX3_Link::timeStep(ctx, local_id);
    }
}

__device__ void VX3_VoxelyzeKernel::updateVoxels(Vindex local_id) {
    if (local_id < ctx.voxels.size()) {
        Vindex material = V_G(local_id, voxel_material);
        if (VM_G(material, fixed))
            return; // fixed voxels, no need to update position
        VX3_Voxel::timeStep(*this, local_id, dt, time);
    }
}

__device__ void VX3_VoxelyzeKernel::updateVoxelTemperature(Vindex local_id) {
    // updates the temperatures For Actuation!
    if (enable_vary_temp and temp_period > 0 and local_id < ctx.voxels.size()) {
        Vindex material = V_G(local_id, voxel_material);
        if (VM_G(material, fixed))
            return; // fixed voxels, no need to update temperature

        Vfloat amplitude = V_G(local_id, amplitude);
        Vfloat frequency = V_G(local_id, frequency);
        Vfloat phase_offset = V_G(local_id, phase_offset);

        Vfloat currentTemperature =
            temp_amplitude * amplitude *
            sin(2 * 3.1415926f * frequency * (time / temp_period + phase_offset));

        if (isnan(currentTemperature)) {
            printf("UVT %f %f %f %f %f %f\n", temp_amplitude, amplitude, frequency, time,
                   temp_period, phase_offset);
        }
        VX3_Voxel::updateTemperature(ctx, local_id, currentTemperature);
    }
}

__device__ void VX3_VoxelyzeKernel::saveRecordFrame(Vindex local_id) {
    unsigned int frame = frame_num;
    Vfloat scale = 1 / rescale;

    if (local_id == 0) {
        d_steps[frame] = step;
        d_time_points[frame] = time;
    }

    if (local_id < ctx.voxels.size() and record_voxel) {
        size_t offset = ctx.voxels.size() * frame;
        VX3_SimulationVoxelRecord v;
        v.valid = VX3_Voxel::isSurface(ctx, local_id);
        if (v.valid) {
            v.material = V_G(local_id, voxel_material);
            v.local_signal = V_G(local_id, local_signal);
            auto position = V_G(local_id, position);
            v.x = position.x * scale;
            v.y = position.y * scale;
            v.z = position.z * scale;
            auto orientation = V_G(local_id, orientation);
            v.orient_angle = orientation.angleDegrees();
            v.orient_x = orientation.x;
            v.orient_y = orientation.y;
            v.orient_z = orientation.z;
            Vec3f ppp = V_G(local_id, ppp_offset), nnn = V_G(local_id, nnn_offset);
            v.nnn_x = nnn.x * scale;
            v.nnn_y = nnn.y * scale;
            v.nnn_z = nnn.z * scale;
            v.ppp_x = ppp.x * scale;
            v.ppp_y = ppp.y * scale;
            v.ppp_z = ppp.z * scale;
        }
        d_voxel_record[offset + local_id] = v;
    }
    if (local_id < ctx.links.size() and record_link) {
        size_t offset = ctx.links.size() * frame;
        VX3_SimulationLinkRecord l;
        l.valid = not L_G(local_id, is_detached);
        if (l.valid) {
            auto pos_voxel_position = V_G(L_G(local_id, voxel_pos), position);
            auto neg_voxel_position = V_G(L_G(local_id, voxel_neg), position);
            l.pos_x = pos_voxel_position.x;
            l.pos_y = pos_voxel_position.y;
            l.pos_z = pos_voxel_position.z;
            l.neg_x = neg_voxel_position.x;
            l.neg_y = neg_voxel_position.y;
            l.neg_z = neg_voxel_position.z;
        }
        d_link_record[offset + local_id] = l;
    }
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::recommendedTimeStep
 *****************************************************************************/
__global__ void computeLinkFreq(VX3_VoxelyzeKernel *kernels, GroupSizesPrefixSum p_sum,
                                GroupToKernelIndex k_index, Vsize group_num,
                                Vfloat *link_freq) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, p_sum.sums, group_num);
    if (gt.gid == NULL_INDEX)
        return;
    Vindex kid = k_index.index[gt.gid];
    auto &ctx = kernels[kid].ctx;

    if (gt.tid < ctx.links.size()) {
        Vindex voxel_neg = L_G(gt.tid, voxel_neg);
        Vindex voxel_pos = L_G(gt.tid, voxel_pos);
        Vindex voxel_neg_mat = V_G(voxel_neg, voxel_material);
        Vindex voxel_pos_mat = V_G(voxel_pos, voxel_material);
        Vfloat mass_neg = VM_G(voxel_neg_mat, mass);
        Vfloat mass_pos = VM_G(voxel_pos_mat, mass);
        Vfloat stiffness = VX3_Link::axialStiffness(ctx, gt.tid);
        // axial
        Vfloat freq = stiffness / (mass_neg < mass_pos ? mass_neg : mass_pos);
        link_freq[tid] = freq;
    }
}

__global__ void computeVoxelFreq(VX3_VoxelyzeKernel *kernels, GroupSizesPrefixSum p_sum,
                                 GroupToKernelIndex k_index, Vsize group_num,
                                 Vfloat *voxel_freq) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, p_sum.sums, group_num);
    if (gt.gid == NULL_INDEX)
        return;
    Vindex kid = k_index.index[gt.gid];
    auto &ctx = kernels[kid].ctx;

    if (tid < ctx.voxels.size()) {
        Vindex mat = V_G(gt.tid, voxel_material);
        Vfloat youngs_modulus = VM_G(mat, E);
        Vfloat nom_size = VM_G(mat, nom_size);
        Vfloat mass = VM_G(mat, mass);
        Vfloat freq = youngs_modulus * nom_size / mass;
        assert(not isnan(freq));
        voxel_freq[tid] = freq;
    }
}

vector<Vfloat> VX3_VoxelyzeKernelBatchExecutor::recommendedTimeStep(
    const vector<size_t> &kernel_indices) const {
    // find the largest natural frequency (sqrt(k/m)) that anything in the
    // simulation will experience, then multiply by 2*pi and invert to get the
    // optimally largest timestep that should retain stability

    // maximum frequency in the simulation in rad/sec
    vector<Vfloat> max_freq(kernel_indices.size(), 0);
    runFunctionAndReduce<MaxReduce<Vfloat>>(computeLinkFreq, max_freq, VF(0), *this, true,
                                            kernel_indices);

    // If link frequency is not available, use voxel frequency
    vector<size_t> invalid_kernel_indices;
    vector<size_t> invalid_kernel_relative_indices;
    for (size_t i = 0; i < kernel_indices.size(); i++) {
        if (max_freq[i] <= VF(0)) {
            invalid_kernel_indices.push_back(kernel_indices[i]);
            invalid_kernel_relative_indices.push_back(i);
        }
    }

    vector<Vfloat> voxel_max_freq(invalid_kernel_indices.size(), 0);
    runFunctionAndReduce<MaxReduce<Vfloat>>(computeVoxelFreq, voxel_max_freq, VF(0),
                                            *this, false, invalid_kernel_indices);

    for (size_t i = 0; i < invalid_kernel_indices.size(); i++)
        max_freq[invalid_kernel_relative_indices[i]] = voxel_max_freq[i];

    for (auto &freq : max_freq) {
        if (freq <= VF(0))
            freq = VF(0);
        else {
            // the optimal time-step is to advance one
            // radian of the highest natural frequency
            freq = 1.0f / (6.283185f * sqrt(freq));
        }
    }
    return std::move(max_freq);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::isAnyLinkDiverged
 *****************************************************************************/
__global__ void checkLinkDivergence(VX3_VoxelyzeKernel *kernels,
                                    GroupSizesPrefixSum p_sum, GroupToKernelIndex k_index,
                                    Vsize group_num, bool *link_diverged) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gt = binary_group_search(tid, p_sum.sums, group_num);
    if (gt.gid == NULL_INDEX)
        return;
    Vindex kid = k_index.index[gt.gid];
    auto &ctx = kernels[kid].ctx;

    if (gt.tid < ctx.links.size()) {
        link_diverged[tid] = L_G(gt.tid, strain) > 100;
    }
}

vector<bool> VX3_VoxelyzeKernelBatchExecutor::isAnyLinkDiverged(
    const std::vector<size_t> &kernel_indices) const {
    vector<bool> result(kernel_indices.size(), false);

    runFunctionAndReduce<OrReduce<bool>>(checkLinkDivergence, result, false, *this, true,
                                         kernel_indices);
    return std::move(result);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::computeCurrentCenterOfMass
 *****************************************************************************/
struct MassDotPos {
    Vfloat dot_x, dot_y, dot_z, mass;
    __host__ __device__ MassDotPos(Vfloat dot_x, Vfloat dot_y, Vfloat dot_z, Vfloat mass)
        : dot_x(dot_x), dot_y(dot_y), dot_z(dot_z), mass(mass) {}
    __device__ MassDotPos operator+(const MassDotPos &mdp) const {
        return MassDotPos(dot_x + mdp.dot_x, dot_y + mdp.dot_y, dot_z + mdp.dot_z,
                          mass + mdp.mass);
    }
};
__global__ void computeMassDotPosition(VX3_VoxelyzeKernel *kernels,
                                       GroupSizesPrefixSum p_sum,
                                       GroupToKernelIndex k_index, Vsize group_num,
                                       MassDotPos *mdp_vec) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gt = binary_group_search(tid, p_sum.sums, group_num);
    if (gt.gid == NULL_INDEX)
        return;
    Vindex kid = k_index.index[gt.gid];
    auto &ctx = kernels[kid].ctx;

    if (gt.tid < ctx.voxels.size()) {
        Vindex mat = V_G(gt.tid, voxel_material);
        bool is_measured = VM_G(mat, is_measured);
        if (not is_measured) {
            auto &mdp = mdp_vec[tid];
            mdp.dot_x = 0;
            mdp.dot_y = 0;
            mdp.dot_z = 0;
            mdp.mass = 0;
        } else {
            Vfloat mat_mass = VM_G(mat, mass);
            Vec3f pos = V_G(gt.tid, position);
            Vec3f dot = pos * mat_mass;
            auto &mdp = mdp_vec[tid];
            mdp.dot_x = dot.x;
            mdp.dot_y = dot.y;
            mdp.dot_z = dot.z;
            mdp.mass = mat_mass;
        }
    }
}

vector<Vec3f> VX3_VoxelyzeKernelBatchExecutor::computeCurrentCenterOfMass(
    const vector<size_t> &kernel_indices) const {
    vector<Vec3f> result(kernel_indices.size(), Vec3f(0, 0, 0));
    vector<MassDotPos> reduce_result(kernel_indices.size(), MassDotPos(0, 0, 0, 0));
    runFunctionAndReduce<SumReduce<MassDotPos>>(computeMassDotPosition, reduce_result,
                                                MassDotPos(0, 0, 0, 0), *this, false,
                                                kernel_indices);
    for (size_t i = 0; i < reduce_result.size(); i++) {
        if (reduce_result[i].mass != 0) {
            auto &mdp = reduce_result[i];
            result[i] =
                Vec3f(mdp.dot_x / mdp.mass, mdp.dot_y / mdp.mass, mdp.dot_z / mdp.mass);
        }
    }
    return std::move(result);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::computeTargetCloseness
 *****************************************************************************/
struct TargetCloseness {
    int num_close_pairs;
    Vfloat closeness;
    __host__ __device__ TargetCloseness(int num_close_pairs, Vfloat closeness)
        : num_close_pairs(num_close_pairs), closeness(closeness) {}
    __device__ TargetCloseness operator+(const TargetCloseness &tc) const {
        return TargetCloseness(num_close_pairs + tc.num_close_pairs,
                               closeness + tc.closeness);
    }
};

__global__ void computeTargetDistances(VX3_VoxelyzeKernel *kernels,
                                       GroupSizesPrefixSum p_sum,
                                       GroupToKernelIndex k_index, Vsize group_num,
                                       TargetCloseness *tc_vec) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gt = binary_group_search(tid, p_sum.sums, group_num);
    if (gt.gid == NULL_INDEX)
        return;
    Vindex kid = k_index.index[gt.gid];

    auto &kernel = kernels[kid];
    auto &ctx = kernel.ctx;

    int local_num_close_pairs = 0;
    Vfloat local_closeness = 0;
    Vfloat radius = kernel.max_dist_in_voxel_lengths_to_count_as_pair * kernel.vox_size;
    if (gt.tid < kernel.target_num) {
        Vindex src_voxel = kernel.target_indices[gt.tid];
        for (unsigned int j = gt.tid + 1; j < kernel.target_num; j++) {
            Vec3f pos1 = V_G(src_voxel, position);
            Vec3f pos2 = V_G(kernel.target_indices[j], position);
            Vfloat distance = pos1.dist(pos2);
            local_num_close_pairs += distance < radius;
            local_closeness += 1 / distance;
        }
    }
    tc_vec[tid].num_close_pairs = local_num_close_pairs;
    tc_vec[tid].closeness = local_closeness;
}

vector<pair<int, Vfloat>> VX3_VoxelyzeKernelBatchExecutor::computeTargetCloseness(
    const std::vector<size_t> &kernel_indices) const {
    // this function is called periodically. not very often. once every thousand of
    // steps.
    vector<size_t> needs_compute_kernel_indices;
    vector<Vsize> needs_compute_kernel_ids;
    for (size_t i = 0; i < kernel_indices.size(); i++) {
        if (kernels[kernel_indices[i]]->max_dist_in_voxel_lengths_to_count_as_pair != 0) {
            needs_compute_kernel_indices.push_back(kernel_indices[i]);
            needs_compute_kernel_ids.push_back(i);
        }
    }

    vector<pair<int, Vfloat>> result(kernel_indices.size(), make_pair(0, 0));
    vector<TargetCloseness> reduce_result(needs_compute_kernel_indices.size(),
                                          TargetCloseness(0, 0));

    runFunctionAndReduce<SumReduce<TargetCloseness>>(
        computeTargetDistances, reduce_result, TargetCloseness(0, 0), *this, false,
        needs_compute_kernel_indices);

    for (size_t i = 0; i < reduce_result.size(); i++) {
        result[needs_compute_kernel_ids[i]] =
            make_pair(reduce_result[i].num_close_pairs, reduce_result[i].closeness);
    }
    return std::move(result);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::init
 *****************************************************************************/
void VX3_VoxelyzeKernelBatchExecutor::init(
    const std::vector<VX3_VoxelyzeKernel *> &kernels_, cudaStream_t stream_, Vfloat dt,
    Vfloat rescale) {
    kernels = kernels_;
    stream = stream_;

    // Setup kernel memory
    VcudaMallocHost(&h_kernels, sizeof(VX3_VoxelyzeKernel) * kernels.size());
    VcudaMallocAsync(&d_kernels, sizeof(VX3_VoxelyzeKernel) * kernels.size(), stream);

    // Setup reduce memory
    vector<Vsize> group_elem_num;
    for (auto k : kernels)
        group_elem_num.push_back(MAX(k->ctx.voxels.size(), k->ctx.links.size()));
    Vsize reduce_buffer_size =
        getReduceByGroupBufferSize(sizeof(Vfloat) * 4, group_elem_num);
    VcudaMallocHost(&h_reduce_output, sizeof(Vfloat) * 4 * kernels.size());
    VcudaMallocHost(&h_sizes,
                    sizeof(Vsize) * kernels.size() * 3 * MAX_GROUP_REDUCE_LEVEL);

    VcudaMallocAsync(&d_reduce1, reduce_buffer_size, stream);
    VcudaMallocAsync(&d_reduce2, reduce_buffer_size, stream);
    VcudaMallocAsync(&d_sizes,
                     sizeof(Vsize) * kernels.size() * 3 * MAX_GROUP_REDUCE_LEVEL, stream);

    for (auto k : kernels) {
        k->rescale = rescale;
        k->dt = dt;
        k->recommended_time_step = dt;
    }

    // To compute recommended time step, we need to synchronize
    // the initial version of the kernel
    copyKernelsAsync();

    // Make sure all allocation and copy actions are completed
    // before calling recommendedTimeStep
    VcudaStreamSynchronize(stream);

    vector<size_t> kernel_indices;
    for (size_t i = 0; i < kernels.size(); i++)
        kernel_indices.push_back(i);

    if (dt <= 0) {
        is_dt_dynamic = true;
        auto recommended_time_steps = recommendedTimeStep(kernel_indices);
        for (size_t i = 0; i < kernel_indices.size(); i++) {
            if (recommended_time_steps[i] < 1e-10) {
                throw std::runtime_error(format(
                    "Recommended_time_step of kernel {} is zero", kernel_indices[i]));
            }
            auto &k = *kernels[kernel_indices[i]];
            k.recommended_time_step = recommended_time_steps[i];
            k.dt = k.dt_frac * k.recommended_time_step;
        }

        // dt may change dynamically (if calling doTimeStep with dt = -1),
        // only use the first value to set update step size
        for (auto k : kernels)
            k->real_step_size = int(k->record_step_size / (10000.0 * k->dt)) + 1;

        auto new_center_of_mass = computeCurrentCenterOfMass(kernel_indices);
        for (size_t i = 0; i < kernel_indices.size(); i++) {
            auto &k = *kernels[kernel_indices[i]];
            k.initial_center_of_mass = new_center_of_mass[i];
            k.current_center_of_mass = new_center_of_mass[i];
        }
    }

    copyKernelsAsync();
    // Make sure all copy actions are completed
    // before calling doTimeStep
    VcudaStreamSynchronize(stream);
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::free
 *****************************************************************************/
void VX3_VoxelyzeKernelBatchExecutor::free() {
    VcudaFreeHost(h_kernels);
    VcudaFreeHost(h_reduce_output);
    VcudaFreeHost(h_sizes);
    VcudaFreeAsync(d_kernels, stream);
    VcudaFreeAsync(d_reduce1, stream);
    VcudaFreeAsync(d_reduce2, stream);
    VcudaFreeAsync(d_sizes, stream);
    h_kernels = nullptr;
    h_reduce_output = nullptr;
    h_sizes = nullptr;
    d_kernels = nullptr;
    d_reduce1 = nullptr;
    d_reduce2 = nullptr;
    d_sizes = nullptr;

    // Make sure all free actions are completed
    VcudaStreamSynchronize(stream);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::doTimeStep
 *****************************************************************************/
__global__ void update_links(VX3_VoxelyzeKernel *kernels, GroupSizesPrefixSum p_sum,
                             GroupToKernelIndex k_index, Vsize group_num) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, p_sum.sums, group_num);
    if (gt.gid == NULL_INDEX)
        return;
    Vindex kid = k_index.index[gt.gid];

    kernels[kid].updateLinks(gt.tid);
}

__global__ void update_voxels(VX3_VoxelyzeKernel *kernels, GroupSizesPrefixSum p_sum,
                              GroupToKernelIndex k_index, Vsize group_num,
                              bool save_frame) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, p_sum.sums, group_num);
    if (gt.gid == NULL_INDEX)
        return;
    Vindex kid = k_index.index[gt.gid];

    kernels[kid].updateVoxels(gt.tid);
    kernels[kid].updateVoxelTemperature(gt.tid);

    if (save_frame and kernels[kid].record_step_size and
        kernels[kid].step % kernels[kid].real_step_size == 0) {
        kernels[kid].saveRecordFrame(gt.tid);
    }
}

vector<bool> VX3_VoxelyzeKernelBatchExecutor::doTimeStep(
    const std::vector<size_t> &kernel_indices, int dt_update_interval,
    int divergence_check_interval, bool save_frame) {

    if (kernel_indices.empty())
        return {};

    vector<bool> result(kernel_indices.size(), true);

    // Update host side kernel time step settings
    if (is_dt_dynamic) {
        if (step % dt_update_interval == 0) {
            auto recommended_time_steps = recommendedTimeStep(kernel_indices);
            for (size_t i = 0; i < kernel_indices.size(); i++) {
                if (recommended_time_steps[i] < 1e-10) {
                    throw std::runtime_error(format(
                        "Recommended_time_step of kernel {} is zero", kernel_indices[i]));
                }
                auto &k = *kernels[kernel_indices[i]];
                k.recommended_time_step = recommended_time_steps[i];
                k.dt = k.dt_frac * k.recommended_time_step;
            }
        }
    }

    vector<size_t> do_time_step_indices;
    if (step % divergence_check_interval == 0) {
        auto link_diverged = isAnyLinkDiverged(kernel_indices);
        for (size_t i = 0; i < kernel_indices.size(); i++) {
            if (link_diverged[i])
                result[i] = false;
        }
    }
    for (size_t i = 0; i < kernel_indices.size(); i++) {
        if (result[i])
            do_time_step_indices.push_back(kernel_indices[i]);
    }

    // Update host side kernel frame storage
    for (auto idx : do_time_step_indices) {
        auto &k = *kernels[idx];
        bool should_save_frame =
            save_frame and k.record_step_size and k.step % k.real_step_size == 0;
        if (should_save_frame)
            k.adjustRecordFrameStorage(k.frame_num + 1, stream);
    }

    // Synchronize host and device side kernel settings
    copyKernelsAsync();

    // Wait for transmission to complete, if not, host may run too fast
    // and change "step", "time" etc. before transmission is finished.
    // Since host is fast, this means waiting for last update link and
    // update voxel, and this transmission to complete.
    VcudaStreamSynchronize(stream);

    // Batch update for all valid kernels
    runFunction(update_links, *this, true, VX3_VOXELYZE_KERNEL_UPDATE_LINK_BLOCK_SIZE,
                do_time_step_indices);
    runFunction(update_voxels, *this, false, VX3_VOXELYZE_KERNEL_UPDATE_VOXEL_BLOCK_SIZE,
                do_time_step_indices, save_frame);

    // Update metrics
    vector<size_t> update_metrics_indices;
    for (auto idx : do_time_step_indices) {
        auto &k = *kernels[idx];
        bool should_save_frame =
            save_frame and k.record_step_size and k.step % k.real_step_size == 0;
        if (should_save_frame)
            k.frame_num++;

        int cycle_step = FLOOR(k.temp_period, k.dt);
        if (k.step % cycle_step == 0) {
            // Sample at the same time point in the cycle, to avoid the
            // impact of actuation as much as possible.
            update_metrics_indices.push_back(idx);
        }
    }
    updateMetrics(update_metrics_indices);
    for (auto idx : do_time_step_indices) {
        auto &k = *kernels[idx];
        k.step++;
        k.time += k.dt;
    }
    step++;
    return std::move(result);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::updateMetrics
 *****************************************************************************/
void VX3_VoxelyzeKernelBatchExecutor::updateMetrics(
    const vector<size_t> &kernel_indices) const {
    if (kernel_indices.empty())
        return;
    for (auto idx : kernel_indices) {
        auto &k = *kernels[idx];
        k.angle_sample_times++;
        k.current_center_of_mass_history[0] = k.current_center_of_mass_history[1];
        k.current_center_of_mass_history[1] = k.current_center_of_mass;
    }

    auto new_center_of_mass = computeCurrentCenterOfMass(kernel_indices);
    for (size_t i = 0; i < kernel_indices.size(); i++) {
        kernels[kernel_indices[i]]->current_center_of_mass = new_center_of_mass[i];
    }

    for (auto idx : kernel_indices) {
        auto &k = *kernels[idx];
        auto A = k.current_center_of_mass_history[0];
        auto B = k.current_center_of_mass_history[1];
        auto C = k.current_center_of_mass;
        if (B == C || A == B || k.angle_sample_times < 3) {
            // avoid divide by zero, and don't include first two steps
            // where A and B are still 0.
            k.recent_angle = 0;
        } else {
            k.recent_angle = acos((B - A).dot(C - B) / (B.dist(A) * C.dist(B)));
        }
    }

    // Also calculate target_closeness here.
    auto new_target_closeness = computeTargetCloseness(kernel_indices);
    for (size_t i = 0; i < kernel_indices.size(); i++) {
        auto &k = *kernels[kernel_indices[i]];
        k.num_close_pairs = new_target_closeness[i].first;
        k.target_closeness = new_target_closeness[i].second;
    }
}

void VX3_VoxelyzeKernelBatchExecutor::copyKernelsAsync() {
    // Synchronize host and device side kernel settings
    for (size_t i = 0; i < kernels.size(); i++) {
        memcpy(h_kernels + i, kernels[i], sizeof(VX3_VoxelyzeKernel));
    }
    VcudaMemcpyAsync(d_kernels, h_kernels, sizeof(VX3_VoxelyzeKernel) * kernels.size(),
                     cudaMemcpyHostToDevice, stream);
}