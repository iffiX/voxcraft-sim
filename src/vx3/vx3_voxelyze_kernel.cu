#include "utils/vx3_conf.h"
#include "utils/vx3_reduce.cuh"
#include "vx3/vx3_simulation_record.h"
#include "vx3_voxelyze_kernel.cuh"
#include <algorithm>
#include <fmt/format.h>
#include <iostream>

using namespace std;
using namespace fmt;

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

/**
 * Compute kernel prefix sum in place.
 * @param kernel_sizes_to_prefix_sum
 *  Stores the number of threads used for each kernel,
 *  after computation it stores prefix sums.
 * @param kernel_num
 *  The number of kernels.
 */
void computeKernelPrefixSum(Vsize *kernel_sizes_to_prefix_sum, Vsize kernel_num) {
    for (Vsize i = 1; i < kernel_num; i++) {
        kernel_sizes_to_prefix_sum[i] =
            kernel_sizes_to_prefix_sum[i] + kernel_sizes_to_prefix_sum[i - 1];
    }
}

/**
 * Run device kernel.
 * @param func CUDA kernel.
 * @param exec Executor.
 * @param is_link Whether the kernel is applied link-wise or voxel-wise.
 * @param block_size Manual configuration of block size for launching kernels.
 * @param kernel_indices The kernel indices to run.
 * @param args Extra arguments for kernel
 */
template <typename FuncType, typename... Args>
void runFunction(FuncType func, const VX3_VoxelyzeKernelBatchExecutor &exec,
                 int block_size, Args &&...args) {

    pair<int, int> size;
    if (block_size > 0) {
        size.first = CEIL(exec.total_threads, block_size);
        size.second = block_size;
    } else
        size = getGridAndBlockSize(func, exec.total_threads);
    func<<<size.first, size.second, 0, exec.stream>>>(
        exec.d_kernels, exec.d_kernel_prefix_sums, exec.d_kernel_is_running,
        exec.kernel_num, std::forward<Args>(args)...);
    CUDA_CHECK_AFTER_CALL();
}

/**
 * Note: runFunctionAndReduce implicitly synchronizes stream before returning results
 * due to the call to reduce_by_group. Reduction is performed for all kernels
 * (including not running ones), but their results will be discarded before returning
 */
template <typename ReduceOp, typename FuncType, typename ResultType>
vector<ResultType> runFunctionAndReduce(FuncType func, ResultType init_value,
                                        const VX3_VoxelyzeKernelBatchExecutor &exec) {
    if (none_of(exec.h_kernel_is_running, exec.h_kernel_is_running + exec.kernel_num,
                [](bool x) { return x; }))
        return {};
    runFunction(func, exec, -1, (ResultType *)exec.d_reduce1);

    //        // For debugging`
    //        Vsize kernel_num = exec.kernel_num;
    //        VcudaStreamSynchronize(exec.stream);
    //        ResultType *tmp;
    //        VcudaMallocHost(&tmp, sizeof(ResultType) *
    //        exec.h_kernel_prefix_sums[kernel_num
    //        - 1]); VcudaMemcpyAsync(tmp, exec.d_reduce1,
    //                         sizeof(ResultType) * exec.h_kernel_prefix_sums[kernel_num -
    //                         1], cudaMemcpyDeviceToHost, exec.stream);
    //        VcudaStreamSynchronize(exec.stream);
    //        size_t offset = 0;
    //        for (size_t k = 0; k < kernel_num; k++) {
    //            std::cout << "Kernel " << k << ": ";
    //            for (size_t i = offset; i < exec.h_kernel_prefix_sums[k]; i++) {
    //                std::cout << ((float *)tmp)[i] << " ";
    //            }
    //            std::cout << std::endl;
    //            offset = exec.h_kernel_prefix_sums[k];
    //        }

    auto partial_result = reduce_by_group<ResultType, ReduceOp>(
        exec.h_reduce_output, exec.d_reduce1, exec.d_reduce2, exec.d_reduce1,
        exec.h_sizes, exec.d_sizes, exec.kernel_len, exec.stream, init_value);

    //        VcudaFreeHost(tmp);

    CUDA_CHECK_AFTER_CALL();
    vector<ResultType> result;
    result.reserve(exec.kernel_num);
    for (size_t i = 0; i < exec.kernel_num; i++) {
        auto k_idx = exec.kernel_relative_indices[i];
        if (exec.h_kernel_is_running[k_idx])
            result.push_back(partial_result[i]);
    }
    return std::move(result);
}

template <typename FuncType, typename... Args>
void runKernelUpdate(FuncType func, const VX3_VoxelyzeKernelBatchExecutor &exec,
                     Args &&...args) {
    auto size = getGridAndBlockSize(func, exec.kernel_num);
    func<<<size.first, size.second, 0, exec.stream>>>(
        exec.d_kernels, exec.d_kernel_is_running, exec.kernel_num,
        std::forward<Args>(args)...);
    CUDA_CHECK_AFTER_CALL();
}
/*****************************************************************************
 * VX3_VoxelyzeKernel::isStopConditionMet
 *****************************************************************************/
bool VX3_VoxelyzeKernel::isStopConditionMet() const {
    if (stop_condition[0].op == mtEND)
        return false;
    else
        return VX3_MathTree::eval(current_center_of_mass.x, current_center_of_mass.y,
                                  current_center_of_mass.z, (Vfloat)collision_count, time,
                                  recent_angle, target_closeness, num_close_pairs,
                                  (int)ctx.voxels.size(), stop_condition) > 0;
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::isResultStartConditionMet
 *****************************************************************************/
bool VX3_VoxelyzeKernel::isResultStartConditionMet() const {
    // By default, when result start condition is empty
    // start recording at the first simulation step
    if (result_start_condition[0].op == mtEND)
        return true;
    else
        return VX3_MathTree::eval(current_center_of_mass.x, current_center_of_mass.y,
                                  current_center_of_mass.z, (Vfloat)collision_count, time,
                                  recent_angle, target_closeness, num_close_pairs,
                                  (int)ctx.voxels.size(), result_start_condition) > 0;
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::isResultEndConditionMet
 *****************************************************************************/
bool VX3_VoxelyzeKernel::isResultEndConditionMet() const {
    // By default, when result stop condition is empty
    // stop recording at the last simulation step
    if (result_end_condition[0].op == mtEND)
        return isStopConditionMet();
    else
        return VX3_MathTree::eval(current_center_of_mass.x, current_center_of_mass.y,
                                  current_center_of_mass.z, (Vfloat)collision_count, time,
                                  recent_angle, target_closeness, num_close_pairs,
                                  (int)ctx.voxels.size(), result_end_condition) > 0;
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::computeFitness
 *****************************************************************************/
Vfloat VX3_VoxelyzeKernel::computeFitness(const Vec3f &start_center_of_mass,
                                          const Vec3f &end_center_of_mass) const {
    Vec3f offset = end_center_of_mass - start_center_of_mass;
    return VX3_MathTree::eval(offset.x, offset.y, offset.z, (Vfloat)collision_count, time,
                              recent_angle, target_closeness, num_close_pairs,
                              (int)ctx.voxels.size(), fitness_function);
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::adjustRecordFrameStorage
 *****************************************************************************/
bool VX3_VoxelyzeKernel::adjustRecordFrameStorage(size_t required_size,
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
        return true;
    }
    return false;
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

    if (local_id == 0)
        frame_num++;
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::recommendedTimeStep
 *****************************************************************************/
__global__ void computeLinkFreq(VX3_VoxelyzeKernel *kernels, const Vsize *p_sum,
                                const bool *k_is_running, Vsize kernel_num,
                                Vfloat *link_freq) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, p_sum, kernel_num);
    if (gt.gid == NULL_INDEX)
        return;
    auto &ctx = kernels[gt.gid].ctx;

    if (k_is_running[gt.gid] and gt.gtid < ctx.links.size()) {
        Vindex voxel_neg = L_G(gt.gtid, voxel_neg);
        Vindex voxel_pos = L_G(gt.gtid, voxel_pos);
        Vindex voxel_neg_mat = V_G(voxel_neg, voxel_material);
        Vindex voxel_pos_mat = V_G(voxel_pos, voxel_material);
        Vfloat mass_neg = VM_G(voxel_neg_mat, mass);
        Vfloat mass_pos = VM_G(voxel_pos_mat, mass);
        Vfloat stiffness = VX3_Link::axialStiffness(ctx, gt.gtid);
        // axial
        Vfloat freq = stiffness / (mass_neg < mass_pos ? mass_neg : mass_pos);
        link_freq[tid] = freq;
    } else
        link_freq[tid] = 0;
}

__global__ void computeVoxelFreq(VX3_VoxelyzeKernel *kernels, const Vsize *p_sum,
                                 const bool *k_is_running, Vsize kernel_num,
                                 Vfloat *voxel_freq) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, p_sum, kernel_num);
    if (gt.gid == NULL_INDEX)
        return;
    auto &ctx = kernels[gt.gid].ctx;

    if (k_is_running[gt.gid] and gt.gtid < ctx.voxels.size()) {
        Vindex mat = V_G(gt.gtid, voxel_material);
        Vfloat youngs_modulus = VM_G(mat, E);
        Vfloat nom_size = VM_G(mat, nom_size);
        Vfloat mass = VM_G(mat, mass);
        Vfloat freq = youngs_modulus * nom_size / mass;
        assert(not isnan(freq));
        voxel_freq[tid] = freq;
    } else
        voxel_freq[tid] = 0;
}

vector<Vfloat> VX3_VoxelyzeKernelBatchExecutor::recommendedTimeStep() const {
    // Compute frequency for running kernels

    // find the largest natural frequency (sqrt(k/m)) that anything in the
    // simulation will experience, then multiply by 2*pi and invert to get the
    // optimally largest timestep that should retain stability

    // maximum frequency in the simulation in rad/sec
    auto link_max_freq =
        runFunctionAndReduce<MaxReduce<Vfloat>>(computeLinkFreq, VF(0), *this);

    auto voxel_max_freq =
        runFunctionAndReduce<MaxReduce<Vfloat>>(computeVoxelFreq, VF(0), *this);

    vector<Vfloat> max_freq;
    for (size_t i = 0; i < link_max_freq.size(); i++)
        max_freq.push_back(link_max_freq[i] == 0 ? voxel_max_freq[i] : link_max_freq[i]);

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
__global__ void checkLinkDivergence(VX3_VoxelyzeKernel *kernels, const Vsize *p_sum,
                                    const bool *k_is_running, Vsize kernel_num,
                                    bool *link_diverged) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gt = binary_group_search(tid, p_sum, kernel_num);
    if (gt.gid == NULL_INDEX)
        return;
    auto &ctx = kernels[gt.gid].ctx;

    if (k_is_running[gt.gid] and gt.gtid < ctx.links.size()) {
        link_diverged[tid] = L_G(gt.gtid, strain) > 100;
    } else
        link_diverged[tid] = false;
}

vector<bool> VX3_VoxelyzeKernelBatchExecutor::isAnyLinkDiverged() const {
    return std::move(
        runFunctionAndReduce<OrReduce<bool>>(checkLinkDivergence, false, *this));
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
__global__ void computeMassDotPosition(VX3_VoxelyzeKernel *kernels, const Vsize *p_sum,
                                       const bool *k_is_running, Vsize kernel_num,
                                       MassDotPos *mdp_vec) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gt = binary_group_search(tid, p_sum, kernel_num);
    if (gt.gid == NULL_INDEX)
        return;
    auto &ctx = kernels[gt.gid].ctx;

    if (k_is_running[gt.gid] and gt.gtid < ctx.voxels.size()) {
        Vindex mat = V_G(gt.gtid, voxel_material);
        bool is_measured = VM_G(mat, is_measured);
        if (not is_measured) {
            auto &mdp = mdp_vec[tid];
            mdp.dot_x = 0;
            mdp.dot_y = 0;
            mdp.dot_z = 0;
            mdp.mass = 0;
        } else {
            Vfloat mat_mass = VM_G(mat, mass);
            Vec3f pos = V_G(gt.gtid, position);
            Vec3f dot = pos * mat_mass;
            auto &mdp = mdp_vec[tid];
            mdp.dot_x = dot.x;
            mdp.dot_y = dot.y;
            mdp.dot_z = dot.z;
            mdp.mass = mat_mass;
        }
    } else {
        auto &mdp = mdp_vec[tid];
        mdp.dot_x = 0;
        mdp.dot_y = 0;
        mdp.dot_z = 0;
        mdp.mass = 0;
    }
}

vector<Vec3f> VX3_VoxelyzeKernelBatchExecutor::computeCurrentCenterOfMass() const {
    vector<Vec3f> result;
    auto reduce_result = runFunctionAndReduce<SumReduce<MassDotPos>>(
        computeMassDotPosition, MassDotPos(0, 0, 0, 0), *this);
    for (auto &mdp : reduce_result) {
        if (mdp.mass != 0) {
            result.emplace_back(mdp.dot_x / mdp.mass, mdp.dot_y / mdp.mass,
                                mdp.dot_z / mdp.mass);
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

__global__ void computeTargetDistances(VX3_VoxelyzeKernel *kernels, const Vsize *p_sum,
                                       const bool *k_is_running, Vsize kernel_num,
                                       TargetCloseness *tc_vec) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto gt = binary_group_search(tid, p_sum, kernel_num);
    if (gt.gid == NULL_INDEX)
        return;
    Vindex kid = gt.gid;
    auto &kernel = kernels[kid];
    auto &ctx = kernel.ctx;

    int local_num_close_pairs = 0;
    Vfloat local_closeness = 0;
    Vfloat radius = kernel.max_dist_in_voxel_lengths_to_count_as_pair * kernel.vox_size;
    if (k_is_running[kid] and gt.gtid < kernel.target_num) {
        Vindex src_voxel = kernel.d_target_indices[gt.gtid];
        for (unsigned int j = gt.gtid + 1; j < kernel.target_num; j++) {
            Vec3f pos1 = V_G(src_voxel, position);
            Vec3f pos2 = V_G(kernel.d_target_indices[j], position);
            Vfloat distance = pos1.dist(pos2);
            local_num_close_pairs += distance < radius;
            local_closeness += 1 / distance;
        }
        tc_vec[tid].num_close_pairs = local_num_close_pairs;
        tc_vec[tid].closeness = local_closeness;
    } else {
        tc_vec[tid].num_close_pairs = 0;
        tc_vec[tid].closeness = 0;
    }
}

vector<pair<int, Vfloat>>
VX3_VoxelyzeKernelBatchExecutor::computeTargetCloseness() const {
    // this function is called periodically. not very often. once every thousand of
    // steps.
    vector<pair<int, Vfloat>> result;

    auto reduce_result = runFunctionAndReduce<SumReduce<TargetCloseness>>(
        computeTargetDistances, TargetCloseness(0, 0), *this);

    for (auto &td : reduce_result) {
        result.emplace_back(td.num_close_pairs, td.closeness);
    }
    return std::move(result);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::init
 *****************************************************************************/
void VX3_VoxelyzeKernelBatchExecutor::init(
    const std::vector<VX3_VoxelyzeKernel *> &kernels_, cudaStream_t stream_, Vfloat dt,
    Vfloat rescale) {
    kernel_num = kernels_.size();
    if (kernel_num == 0)
        return;
    stream = stream_;

    // Setup kernel memory
    VcudaMallocHost(&h_kernels, sizeof(VX3_VoxelyzeKernel) * kernel_num);
    VcudaMallocAsync(&d_kernels, sizeof(VX3_VoxelyzeKernel) * kernel_num, stream);
    for (size_t k_idx = 0; k_idx < kernel_num; k_idx++) {
        memcpy(h_kernels + k_idx, kernels_[k_idx], sizeof(VX3_VoxelyzeKernel));
    }

    // Setup synchronization memory
    VcudaMallocHost(&h_kernel_sync, sizeof(VX3_VoxelyzeKernel) * kernel_num);
    VcudaMallocAsync(&d_kernel_sync, sizeof(VX3_VoxelyzeKernel) * kernel_num, stream);

    // Setup kernel location info memory
    VcudaMallocHost(&h_kernel_prefix_sums, sizeof(Vsize) * kernel_num);
    VcudaMallocHost(&h_kernel_is_running, sizeof(bool) * kernel_num);
    VcudaMallocAsync(&d_kernel_prefix_sums, sizeof(Vsize) * kernel_num, stream);
    VcudaMallocAsync(&d_kernel_is_running, sizeof(bool) * kernel_num, stream);

    // Setup reduce memory
    vector<Vsize> group_elem_num;
    for (Vsize k_idx = 0; k_idx < kernel_num; k_idx++) {
        auto &k = h_kernels[k_idx];
        group_elem_num.push_back(MAX(k.ctx.voxels.size(), k.ctx.links.size()));
    }

    Vsize reduce_buffer_size =
        getReduceByGroupBufferSize(sizeof(Vfloat) * 4, group_elem_num);
    VcudaMallocHost(&h_reduce_output, sizeof(Vfloat) * 4 * kernel_num);
    VcudaMallocHost(&h_sizes, sizeof(Vsize) * kernel_num * 3 * MAX_GROUP_REDUCE_LEVEL);
    VcudaMallocAsync(&d_reduce1, reduce_buffer_size, stream);
    VcudaMallocAsync(&d_reduce2, reduce_buffer_size, stream);
    VcudaMallocAsync(&d_sizes, sizeof(Vsize) * kernel_num * 3 * MAX_GROUP_REDUCE_LEVEL,
                     stream);

    // Setup data for computing recommended time step
    for (Vsize k_idx = 0; k_idx < kernel_num; k_idx++) {
        auto &k = h_kernels[k_idx];
        k.rescale = rescale;
        k.dt = dt;
        k.recommended_time_step = dt;
    }

    // To compute recommended time step, we need to synchronize
    // the partially initialized kernel
    VcudaMemcpyAsync(d_kernels, h_kernels, sizeof(VX3_VoxelyzeKernel) * kernel_num,
                     cudaMemcpyHostToDevice, stream);

    // Then we need to initialize location info for running threads, and enable all
    // kernels
    initThreadRunInfo();
    updateKernelIsRunning({}, true);
    // Make sure all allocation and copy actions are completed
    // before calling recommendedTimeStep
    VcudaStreamSynchronize(stream);

    if (dt <= 0) {
        is_dt_dynamic = true;
        auto recommended_time_steps = recommendedTimeStep();
        for (size_t k_idx = 0; k_idx < kernel_num; k_idx++) {
            if (recommended_time_steps[k_idx] < 1e-10) {
                throw std::runtime_error(
                    format("Recommended_time_step of kernel {} is zero", k_idx));
            }
            auto &k = h_kernels[k_idx];
            k.recommended_time_step = recommended_time_steps[k_idx];
            k.dt = k.dt_frac * k.recommended_time_step;
        }

        // dt may change dynamically (if calling doTimeStep with dt = -1),
        // only use the first value to set update step size
        for (Vsize k_idx = 0; k_idx < kernel_num; k_idx++) {
            auto &k = h_kernels[k_idx];
            k.real_step_size = int(k.record_step_size / (10000.0 * k.dt)) + 1;
        }

        auto new_center_of_mass = computeCurrentCenterOfMass();
        for (Vsize k_idx = 0; k_idx < kernel_num; k_idx++) {
            auto &k = h_kernels[k_idx];
            k.initial_center_of_mass = new_center_of_mass[k_idx];
            k.current_center_of_mass = new_center_of_mass[k_idx];
        }
    }

    // Synchronize the fully initialized kernel
    VcudaMemcpyAsync(d_kernels, h_kernels, sizeof(VX3_VoxelyzeKernel) * kernel_num,
                     cudaMemcpyHostToDevice, stream);

    // Make sure all copy actions are completed
    // before calling doTimeStep
    VcudaStreamSynchronize(stream);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::initThreadRunInfo
 *****************************************************************************/
void VX3_VoxelyzeKernelBatchExecutor::initThreadRunInfo() {
    Vsize relative_idx = 0;

    kernel_len.clear();
    kernel_relative_indices.clear();
    // Compute thread location info (prefix sum, kernel index) for running kernel
    for (Vsize k_idx = 0; k_idx < kernel_num; k_idx++) {
        size_t link_num = (h_kernels + k_idx)->ctx.links.size();
        size_t voxel_num = (h_kernels + k_idx)->ctx.voxels.size();
        if (link_num > 0 or voxel_num > 0) {
            size_t size = kernel_relative_indices.size();
            h_kernel_prefix_sums[size] = MAX(link_num, voxel_num);
            kernel_len.push_back(link_num);
            kernel_relative_indices.push_back(relative_idx);
        }
        relative_idx++;
    }

    computeKernelPrefixSum(h_kernel_prefix_sums, kernel_num);
    total_threads = h_kernel_prefix_sums[kernel_num - 1];

    // Copy thread location info to device
    VcudaMemcpyAsync(d_kernel_prefix_sums, h_kernel_prefix_sums,
                     sizeof(Vsize) * kernel_num, cudaMemcpyHostToDevice, stream);
    VcudaMemcpyAsync(d_kernel_is_running, h_kernel_is_running, sizeof(bool) * kernel_num,
                     cudaMemcpyHostToDevice, stream);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::updateKernelIsRunning
 *****************************************************************************/
void VX3_VoxelyzeKernelBatchExecutor::updateKernelIsRunning(
    const vector<size_t> &kernel_indices, bool init) {
    if (not init) {
        auto new_h_kernel_is_running = new bool[kernel_num];
        memset(new_h_kernel_is_running, false, sizeof(bool) * kernel_num);
        for (auto k_idx : kernel_indices) {
            new_h_kernel_is_running[k_idx] = true;
        }
        if (memcmp(new_h_kernel_is_running, h_kernel_is_running,
                   sizeof(bool) * kernel_num) != 0) {
            memcpy(h_kernel_is_running, new_h_kernel_is_running,
                   sizeof(bool) * kernel_num);
            VcudaMemcpyAsync(d_kernel_is_running, h_kernel_is_running,
                             sizeof(bool) * kernel_num, cudaMemcpyHostToDevice, stream);
        }
        delete[] new_h_kernel_is_running;
    } else {
        memset(h_kernel_is_running, true, sizeof(bool) * kernel_num);
        VcudaMemcpyAsync(d_kernel_is_running, h_kernel_is_running,
                         sizeof(bool) * kernel_num, cudaMemcpyHostToDevice, stream);
    }
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::free
 *****************************************************************************/
void VX3_VoxelyzeKernelBatchExecutor::free() {
    VcudaFreeHost(h_kernels);
    VcudaFreeHost(h_kernel_sync);
    VcudaFreeHost(h_kernel_prefix_sums);
    VcudaFreeHost(h_kernel_is_running);
    VcudaFreeHost(h_reduce_output);
    VcudaFreeHost(h_sizes);
    VcudaFreeAsync(d_kernels, stream);
    VcudaFreeAsync(d_kernel_sync, stream);
    VcudaFreeAsync(d_kernel_prefix_sums, stream);
    VcudaFreeAsync(d_kernel_is_running, stream);
    VcudaFreeAsync(d_reduce1, stream);
    VcudaFreeAsync(d_reduce2, stream);
    VcudaFreeAsync(d_sizes, stream);
    h_kernels = nullptr;
    h_kernel_sync = nullptr;
    h_kernel_prefix_sums = nullptr;
    h_kernel_is_running = nullptr;
    h_reduce_output = nullptr;
    h_sizes = nullptr;
    d_kernels = nullptr;
    d_kernel_sync = nullptr;
    d_kernel_prefix_sums = nullptr;
    d_kernel_is_running = nullptr;
    d_reduce1 = nullptr;
    d_reduce2 = nullptr;
    d_sizes = nullptr;

    // Make sure all free actions are completed
    VcudaStreamSynchronize(stream);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::getKernel
 *****************************************************************************/
const VX3_VoxelyzeKernel &
VX3_VoxelyzeKernelBatchExecutor::getKernel(size_t kernel_index) {
    return h_kernels[kernel_index];
}
/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::doTimeStep
 *****************************************************************************/
struct FrameStorageInfo {
    Vsize frame_storage_size;
    unsigned long *d_steps = nullptr;
    Vfloat *d_time_points = nullptr;
    VX3_SimulationLinkRecord *d_link_record = nullptr;
    VX3_SimulationVoxelRecord *d_voxel_record = nullptr;
};

__global__ void update_links(VX3_VoxelyzeKernel *kernels, const Vsize *p_sum,
                             const bool *k_is_running, Vsize kernel_num) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, p_sum, kernel_num);
    if (gt.gid == NULL_INDEX or not k_is_running[gt.gid])
        return;

    kernels[gt.gid].updateLinks(gt.gtid);
}

__global__ void update_voxels(VX3_VoxelyzeKernel *kernels, const Vsize *p_sum,
                              const bool *k_is_running, Vsize kernel_num,
                              bool save_frame) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto gt = binary_group_search(tid, p_sum, kernel_num);
    if (gt.gid == NULL_INDEX or not k_is_running[gt.gid])
        return;
    Vindex kid = gt.gid;

    kernels[kid].updateVoxels(gt.gtid);
    kernels[kid].updateVoxelTemperature(gt.gtid);

    if (save_frame and kernels[kid].record_step_size and
        kernels[kid].step % kernels[kid].real_step_size == 0) {
        kernels[kid].saveRecordFrame(gt.gtid);
    }
}

__global__ void update_time_step(VX3_VoxelyzeKernel *kernels, const bool *k_is_running,
                                 Vsize kernel_num, const Vfloat *recommended_time_steps) {
    unsigned int kid = threadIdx.x + blockIdx.x * blockDim.x;
    if (kid < kernel_num and k_is_running[kid]) {
        auto &k = kernels[kid];
        k.recommended_time_step = recommended_time_steps[kid];
        k.dt = k.dt_frac * k.recommended_time_step;
    }
}

__global__ void update_frame_storage(VX3_VoxelyzeKernel *kernels,
                                     const bool *k_is_running, Vsize kernel_num,
                                     const FrameStorageInfo *info) {
    unsigned int kid = threadIdx.x + blockIdx.x * blockDim.x;
    if (kid < kernel_num and k_is_running[kid]) {
        auto &k = kernels[kid];
        k.d_steps = info[kid].d_steps;
        k.d_time_points = info[kid].d_time_points;
        k.d_link_record = info[kid].d_link_record;
        k.d_voxel_record = info[kid].d_voxel_record;
        k.frame_storage_size = info[kid].frame_storage_size;
    }
}

__global__ void update_time(VX3_VoxelyzeKernel *kernels, const bool *k_is_running,
                            Vsize kernel_num) {
    unsigned int kid = threadIdx.x + blockIdx.x * blockDim.x;
    if (kid < kernel_num and k_is_running[kid]) {
        auto &k = kernels[kid];
        k.step++;
        k.time += k.dt;
    }
}

vector<bool> VX3_VoxelyzeKernelBatchExecutor::doTimeStep(
    const std::vector<size_t> &kernel_indices, int dt_update_interval,
    int divergence_check_interval, bool save_frame) {
    if (kernel_indices.empty())
        return {};

    VcudaStreamSynchronize(stream);
    updateKernelIsRunning(kernel_indices);
    vector<bool> result(kernel_indices.size(), true);
    bool state_require_sync = false;
    // Update host side kernel time step settings
    if (is_dt_dynamic) {
        if (step % dt_update_interval == 0) {
            bool recommended_time_steps_updated = false;
            auto recommended_time_steps = recommendedTimeStep();
            for (size_t i = 0; i < kernel_indices.size(); i++) {
                if (recommended_time_steps[i] < 1e-10) {
                    throw std::runtime_error(format(
                        "Recommended_time_step of kernel {} is zero", kernel_indices[i]));
                }
                auto &k = h_kernels[kernel_indices[i]];
                if (k.recommended_time_step != recommended_time_steps[i]) {
                    k.recommended_time_step = recommended_time_steps[i];
                    k.dt = k.dt_frac * k.recommended_time_step;
                    state_require_sync = true;
                    recommended_time_steps_updated = true;
                }
            }
            if (recommended_time_steps_updated) {
                for (size_t i = 0; i < kernel_indices.size(); i++) {
                    ((Vfloat *)h_kernel_sync)[i] =
                        h_kernels[kernel_indices[i]].recommended_time_step;
                }
                cudaMemcpyAsync(d_kernel_sync, h_kernel_sync,
                                sizeof(Vfloat) * kernel_indices.size(),
                                cudaMemcpyHostToDevice, stream);
                runKernelUpdate(update_time_step, *this, (Vfloat *)d_kernel_sync);
            }
        }
    }

    // Update host side kernel frame storage
    bool frame_storage_adjusted = false;
    for (auto k_idx : kernel_indices) {
        auto &k = h_kernels[k_idx];
        bool should_save_frame =
            save_frame and k.record_step_size and k.step % k.real_step_size == 0;
        if (should_save_frame) {
            // ! do not use or, eg: frame_storage_adjusted or k.adjustRecord...
            // because if one kernel has adjusted, frame_storage_adjusted is true
            // then due to lazy evaluation k.adjustRecord... won't be executed
            // for other kernels
            if (k.adjustRecordFrameStorage(k.frame_num + 1, stream))
                frame_storage_adjusted = true;
        }
    }

    if (frame_storage_adjusted) {
        state_require_sync = true;
        size_t pad_bytes = ALIGN(kernel_num * sizeof(Vfloat), sizeof(FrameStorageInfo));
        auto h_frame_update_start =
            (FrameStorageInfo *)((char *)h_kernel_sync + pad_bytes);
        auto d_frame_update_start =
            (FrameStorageInfo *)((char *)d_kernel_sync + pad_bytes);
        for (size_t i = 0; i < kernel_indices.size(); i++) {
            // Make sure this memory region does not overlap with the
            // region used for recommended_time_step update, since memory
            // copy is asynchronous
            auto &pointers = h_frame_update_start[i];
            pointers.d_steps = h_kernels[kernel_indices[i]].d_steps;
            pointers.d_time_points = h_kernels[kernel_indices[i]].d_time_points;
            pointers.d_link_record = h_kernels[kernel_indices[i]].d_link_record;
            pointers.d_voxel_record = h_kernels[kernel_indices[i]].d_voxel_record;
            pointers.frame_storage_size = h_kernels[kernel_indices[i]].frame_storage_size;
        }
        cudaMemcpyAsync(d_frame_update_start, h_frame_update_start,
                        sizeof(FrameStorageInfo) * kernel_indices.size(),
                        cudaMemcpyHostToDevice, stream);
        runKernelUpdate(update_frame_storage, *this, d_frame_update_start);
    }

    // Synchronize host and device side kernel settings
    if (state_require_sync) {
        // Wait for transmission to complete, if not, host may run too fast
        // and change "step", "time" etc. before transmission is finished.
        // Since host is fast, this means waiting for last update link and
        // update voxel, and this transmission to complete.
        VcudaStreamSynchronize(stream);
    }

    // Batch update for all valid kernels
    runFunction(update_links, *this, VX3_VOXELYZE_KERNEL_UPDATE_LINK_BLOCK_SIZE);
    runFunction(update_voxels, *this, VX3_VOXELYZE_KERNEL_UPDATE_VOXEL_BLOCK_SIZE,
                save_frame);

    // Update frame number, and metrics
    unordered_set<size_t> update_metrics_indices;
    for (auto k_idx : kernel_indices) {
        auto &k = h_kernels[k_idx];
        bool should_save_frame =
            save_frame and k.record_step_size and k.step % k.real_step_size == 0;
        if (should_save_frame)
            k.frame_num++;

        int cycle_step = FLOOR(k.temp_period, k.dt);
        if (k.step % cycle_step == 0) {
            // Sample at the same time point in the cycle, to avoid the
            // impact of actuation as much as possible.
            update_metrics_indices.insert(k_idx);
        }
    }
    updateMetrics(update_metrics_indices);
    for (auto k_idx : kernel_indices) {
        auto &k = h_kernels[k_idx];
        k.step++;
        k.time += k.dt;
    }
    runKernelUpdate(update_time, *this);
    step++;

    if (step % divergence_check_interval == 0) {
        auto link_diverged = isAnyLinkDiverged();
        for (size_t i = 0; i < kernel_indices.size(); i++) {
            if (link_diverged[i])
                result[i] = false;
        }
    }
    return std::move(result);
}

/*****************************************************************************
 * VX3_VoxelyzeKernelBatchExecutor::updateMetrics
 *****************************************************************************/
void VX3_VoxelyzeKernelBatchExecutor::updateMetrics(
    const unordered_set<size_t> &update_metrics_kernel_indices) const {
    for (auto k_idx : update_metrics_kernel_indices) {
        auto &k = h_kernels[k_idx];
        k.angle_sample_times++;
        k.current_center_of_mass_history[0] = k.current_center_of_mass_history[1];
        k.current_center_of_mass_history[1] = k.current_center_of_mass;
    }
    if (not update_metrics_kernel_indices.empty()) {
        auto new_center_of_mass = computeCurrentCenterOfMass();
        for (Vsize k_idx = 0, i = 0; k_idx < kernel_num; k_idx++) {
            if (h_kernel_is_running[k_idx]) {
                if (update_metrics_kernel_indices.find(k_idx) !=
                    update_metrics_kernel_indices.end()) {
                    auto &k = h_kernels[k_idx];
                    k.current_center_of_mass = new_center_of_mass[i];
                    auto A = k.current_center_of_mass_history[0];
                    auto B = k.current_center_of_mass_history[1];
                    auto C = k.current_center_of_mass;
                    if (B == C || A == B || k.angle_sample_times < 3) {
                        // avoid divide by zero, and don't include first two steps
                        // where A and B are still 0.
                        k.recent_angle = 0;
                    } else {
                        k.recent_angle =
                            acos((B - A).dot(C - B) / (B.dist(A) * C.dist(B)));
                    }
                }
                i++;
            }
        }
        // Also calculate target_closeness here.
        auto new_target_closeness = computeTargetCloseness();
        for (Vsize k_idx = 0, i = 0; k_idx < kernel_num; k_idx++) {
            if (h_kernel_is_running[k_idx]) {
                if (update_metrics_kernel_indices.find(k_idx) !=
                    update_metrics_kernel_indices.end()) {
                    auto &k = h_kernels[k_idx];
                    k.num_close_pairs = new_target_closeness[i].first;
                    k.target_closeness = new_target_closeness[i].second;
                }
                i++;
            }
        }
    }
}