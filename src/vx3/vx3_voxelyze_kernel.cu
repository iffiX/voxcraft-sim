#include "utils/vx3_conf.h"
#include "utils/vx3_reduce.cuh"
#include "vx3/vx3_simulation_record.h"
#include "vx3_voxelyze_kernel.cuh"
#include <iostream>

#define ALLOCATE_FRAME_NUM 10000

/* Tools */
template <typename T>
inline std::pair<int, int> getGridAndBlockSize(T func, size_t item_num) {
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
    return std::make_pair(grid_size, block_size);
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::recommendedTimeStep
 *****************************************************************************/
__global__ void computeLinkFreq(VX3_Context ctx, Vfloat *link_freq) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ctx.links.size()) {
        Vindex voxel_neg = L_G(tid, voxel_neg);
        Vindex voxel_pos = L_G(tid, voxel_pos);
        Vindex voxel_neg_mat = V_G(voxel_neg, voxel_material);
        Vindex voxel_pos_mat = V_G(voxel_pos, voxel_material);
        Vfloat mass_neg = VM_G(voxel_neg_mat, mass);
        Vfloat mass_pos = VM_G(voxel_pos_mat, mass);
        Vfloat stiffness = VX3_Link::axialStiffness(ctx, tid);
        // axial
        link_freq[tid] = stiffness / (mass_neg < mass_pos ? mass_neg : mass_pos);
    }
}

__global__ void computeVoxelFreq(VX3_Context ctx, Vfloat *voxel_freq) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ctx.voxels.size()) {
        Vindex mat = V_G(tid, voxel_material);
        Vfloat youngs_modulus = VM_G(mat, E);
        Vfloat nom_size = VM_G(mat, nom_size);
        Vfloat mass = VM_G(mat, mass);
        voxel_freq[tid] = youngs_modulus * nom_size / mass;
    }
}

Vfloat VX3_VoxelyzeKernel::recommendedTimeStep() const {
    // find the largest natural frequency (sqrt(k/m)) that anything in the
    // simulation will experience, then multiply by 2*pi and invert to get the
    // optimally largest timestep that should retain stability

    // maximum frequency in the simulation in rad/sec
    Vfloat max_freq = 0;
    if (ctx.links.size() > 0) {
        auto size = getGridAndBlockSize(computeLinkFreq, ctx.links.size());
        computeLinkFreq<<<size.first, size.second, 0, stream>>>(ctx, (Vfloat *)d_reduce1);
        CUDA_CHECK_AFTER_CALL();
        max_freq = reduce<Vfloat, maxReduce<Vfloat>>(
            h_reduce_output, d_reduce1, d_reduce2, d_reduce1, ctx.links.size(), stream);
    }
    if (max_freq <= 0.0f) {
        // didn't find anything (i.e no links) check for
        // individual voxels
        auto size = getGridAndBlockSize(computeVoxelFreq, ctx.voxels.size());
        computeVoxelFreq<<<size.first, size.second, 0, stream>>>(ctx,
                                                                 (Vfloat *)d_reduce1);
        CUDA_CHECK_AFTER_CALL();
        max_freq = reduce<Vfloat, maxReduce<Vfloat>>(
            h_reduce_output, d_reduce1, d_reduce2, d_reduce1, ctx.voxels.size(), stream);
    }

    if (max_freq <= 0.0f)
        return 0.0f;
    else {
        // the optimal time-step is to advance one
        // radian of the highest natural frequency
        return 1.0f / (6.283185f * sqrt(max_freq));
    }
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::stopConditionMet
 *****************************************************************************/
bool VX3_VoxelyzeKernel::isStopConditionMet() const {
    return VX3_MathTree::eval(current_center_of_mass.x, current_center_of_mass.y,
                              current_center_of_mass.z, (Vfloat)collision_count, time,
                              recent_angle, target_closeness, num_close_pairs,
                              (int)ctx.voxels.size(), stop_condition_formula) > 0;
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::isAnyLinkDiverged
 *****************************************************************************/
__global__ void checkLinkDivergence(VX3_Context ctx, bool *link_diverged) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < ctx.links.size()) {
        link_diverged[tid] = L_G(tid, strain) > 100;
    }
}

bool VX3_VoxelyzeKernel::isAnyLinkDiverged() const {
    if (ctx.links.size() > 0) {
        auto size = getGridAndBlockSize(checkLinkDivergence, ctx.links.size());
        checkLinkDivergence<<<size.first, size.second, 0, stream>>>(ctx,
                                                                    (bool *)d_reduce1);
        CUDA_CHECK_AFTER_CALL();
        return reduce<bool, orReduce<bool>>(h_reduce_output, d_reduce1, d_reduce2,
                                            d_reduce1, ctx.links.size(), stream, false);
    } else
        return false;
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
 * VX3_VoxelyzeKernel::computeCurrentCenterOfMass
 *****************************************************************************/
__global__ void computeMassDotPosition(VX3_Context ctx, Vfloat *dot_x, Vfloat *dot_y,
                                       Vfloat *dot_z, Vfloat *mass) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < ctx.voxels.size()) {
        Vindex mat = V_G(tid, voxel_material);
        bool is_measured = VM_G(mat, is_measured);
        if (not is_measured) {
            dot_x[tid] = 0;
            dot_y[tid] = 0;
            dot_z[tid] = 0;
            mass[tid] = 0;
        } else {
            Vfloat mat_mass = VM_G(mat, mass);
            Vec3f pos = V_G(tid, position);
            Vec3f dot = pos * mat_mass;
            dot_x[tid] = dot.x;
            dot_y[tid] = dot.y;
            dot_z[tid] = dot.z;
            mass[tid] = mat_mass;
        }
    }
}

Vec3f VX3_VoxelyzeKernel::computeCurrentCenterOfMass() const {
    size_t v_num = ctx.voxels.size();
    auto start = (Vfloat *)d_reduce1;
    Vfloat *dot_x = start, *dot_y = start + v_num, *dot_z = start + v_num * 2,
           *mass = start + v_num * 3;
    auto size = getGridAndBlockSize(computeMassDotPosition, ctx.voxels.size());
    computeMassDotPosition<<<size.first, size.second, 0, stream>>>(ctx, dot_x, dot_y,
                                                                   dot_z, mass);
    CUDA_CHECK_AFTER_CALL();
    auto dot_x_sum = reduce<Vfloat, sumReduce<Vfloat>>(h_reduce_output, dot_x, d_reduce2,
                                                       dot_x, v_num, stream);
    auto dot_y_sum = reduce<Vfloat, sumReduce<Vfloat>>(h_reduce_output, dot_y, d_reduce2,
                                                       dot_y, v_num, stream);
    auto dot_z_sum = reduce<Vfloat, sumReduce<Vfloat>>(h_reduce_output, dot_z, d_reduce2,
                                                       dot_z, v_num, stream);
    auto mass_sum = reduce<Vfloat, sumReduce<Vfloat>>(h_reduce_output, mass, d_reduce2,
                                                      mass, v_num, stream);
    if (mass_sum == 0)
        return {};
    return {dot_x_sum / mass_sum, dot_y_sum / mass_sum, dot_z_sum / mass_sum};
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::computeTargetCloseness
 *****************************************************************************/
__global__ void computeTargetDistances(VX3_VoxelyzeKernel kernel, Vfloat radius,
                                       int *num_close_pairs, Vfloat *closeness) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_num_close_pairs = 0;
    Vfloat local_closeness = 0;
    auto &ctx = kernel.ctx;
    if (tid < kernel.target_num) {
        Vindex src_voxel = kernel.target_indices[tid];
        for (unsigned int j = tid + 1; j < kernel.target_num; j++) {
            Vec3f pos1 = V_G(src_voxel, position);
            Vec3f pos2 = V_G(kernel.target_indices[j], position);
            Vfloat distance = pos1.dist(pos2);
            local_num_close_pairs += distance < radius;
            local_closeness += 1 / distance;
        }
    }
    num_close_pairs[tid] = local_num_close_pairs;
    closeness[tid] = local_closeness;
}

std::pair<int, Vfloat> VX3_VoxelyzeKernel::computeTargetCloseness() const {
    // this function is called periodically. not very often. once every thousand of
    // steps.
    if (max_dist_in_voxel_lengths_to_count_as_pair == 0)
        return std::make_pair(0, 0);
    Vfloat radius = max_dist_in_voxel_lengths_to_count_as_pair * vox_size;

    auto voxel_num_close_pairs = (int *)d_reduce1;
    auto closeness = (Vfloat *)(voxel_num_close_pairs + target_num);
    auto size = getGridAndBlockSize(computeTargetDistances, target_num);
    computeTargetDistances<<<size.first, size.second, 0, stream>>>(
        *this, radius, voxel_num_close_pairs, closeness);
    int total_close_pairs =
        reduce<int, sumReduce<int>>(h_reduce_output, voxel_num_close_pairs, d_reduce2,
                                    voxel_num_close_pairs, target_num, stream);
    Vfloat total_closeness = reduce<Vfloat, sumReduce<Vfloat>>(
        h_reduce_output, closeness, d_reduce2, closeness, target_num, stream);
    return std::make_pair(total_close_pairs, total_closeness);
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::init
 *****************************************************************************/
void VX3_VoxelyzeKernel::init(Vfloat dt_, Vfloat rescale_) {

    rescale = rescale_;
    dt = dt_;
    recommended_time_step = dt_;
    if (dt_ <= 0) {
        is_dt_dynamic = true;
        recommended_time_step = recommendedTimeStep();
        dt = dt_frac * recommended_time_step;
    }

    // dt may change dynamically (if calling doTimeStep with dt = -1),
    // only use the first value to
    real_step_size = int(record_step_size / (10000.0 * dt)) + 1;
    current_center_of_mass = computeCurrentCenterOfMass();
    initial_center_of_mass = current_center_of_mass;
}

/*****************************************************************************
 * VX3_VoxelyzeKernel::doTimeStep
 *****************************************************************************/
// When the max number of links/voxels is smaller than max allowed block size
// (i.e. fits into one block), use this function to reduce kernel launch cost
__global__ void update_combined(VX3_VoxelyzeKernel kernel, bool save_frame) {
    // FIXME: I changed the order of update and moved voxel temperature
    //  update to the last to avoid confliction with link updates
    kernel.updateLinks();
    __syncthreads();
    kernel.updateVoxels();
    kernel.updateVoxelTemperature();
    // save_frame is used to override record_step_size in the configuration file
    if (save_frame and kernel.record_step_size and
        kernel.step % kernel.real_step_size == 0)
        kernel.saveRecordFrame();
}

__global__ void update_separate_links(VX3_VoxelyzeKernel kernel) { kernel.updateLinks(); }

__global__ void update_separate_voxels(VX3_VoxelyzeKernel kernel, bool save_frame) {
    kernel.updateVoxels();
    kernel.updateVoxelTemperature();
    // save_frame is used to override record_step_size in the configuration file
    if (save_frame and kernel.record_step_size and
        kernel.step % kernel.real_step_size == 0)
        kernel.saveRecordFrame();
}

bool VX3_VoxelyzeKernel::doTimeStep(int dt_update_interval, int divergence_check_interval,
                                    bool save_frame) {
    if (is_dt_dynamic) {
        if (step % dt_update_interval == 0) {
            recommended_time_step = recommendedTimeStep();
            if (recommended_time_step < 1e-10) {
                std::cout << "Warning: recommended_time_step is zero." << std::endl;
                recommended_time_step = 1e-10;
            }
            dt = dt_frac * recommended_time_step;
        }
    }

    if (step % divergence_check_interval == 0 and isAnyLinkDiverged())
        return false;

    // Main update part
    bool should_save_frame =
        save_frame and record_step_size and step % real_step_size == 0;
    if (should_save_frame)
        adjustRecordFrameStorage(frame_num + 1);
    // See if it fits into one block
    auto size =
        getGridAndBlockSize(update_combined, MAX(ctx.voxels.size(), ctx.links.size()));
    if (size.first == 1) {
        update_combined<<<size.first, size.second, 0, stream>>>(*this, save_frame);
    } else {
        size = getGridAndBlockSize(update_separate_links, ctx.links.size());
        update_separate_links<<<size.first, size.second, 0, stream>>>(*this);
        size = getGridAndBlockSize(update_separate_voxels, ctx.voxels.size());
        update_separate_voxels<<<size.first, size.second, 0, stream>>>(*this, save_frame);
    }
    if (should_save_frame)
        frame_num++;

    int cycle_step = FLOOR(temp_period, dt);
    if (step % cycle_step == 0) {
        // Sample at the same time point in the cycle, to avoid the
        // impact of actuation as much as possible.
        updateMetrics();
    }
    step++;
    time += dt;
    return true;
}

void VX3_VoxelyzeKernel::adjustRecordFrameStorage(size_t required_size) {
    if (frame_storage_size < required_size) {
        unsigned long *new_d_steps;
        Vfloat *new_d_time_points;
        VX3_SimulationLinkRecord *new_d_link_record;
        VX3_SimulationVoxelRecord *new_d_voxel_record;
        size_t new_frame_capacity = frame_storage_size + ALLOCATE_FRAME_NUM;
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

void VX3_VoxelyzeKernel::updateMetrics() {
    angle_sample_times++;

    current_center_of_mass_history[0] = current_center_of_mass_history[1];
    current_center_of_mass_history[1] = current_center_of_mass;
    current_center_of_mass = computeCurrentCenterOfMass();
    auto A = current_center_of_mass_history[0];
    auto B = current_center_of_mass_history[1];
    auto C = current_center_of_mass;
    if (B == C || A == B || angle_sample_times < 3) {
        // avoid divide by zero, and don't include first two steps
        // where A and B are still 0.
        recent_angle = 0;
    } else {
        recent_angle = acos((B - A).dot(C - B) / (B.dist(A) * C.dist(B)));
    }
    // Also calculate target_closeness here.
    auto closeness = computeTargetCloseness();
    num_close_pairs = closeness.first;
    target_closeness = closeness.second;
}

__device__ void VX3_VoxelyzeKernel::updateLinks() {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < ctx.links.size()) {
        Vindex voxel_neg = L_G(tid, voxel_neg);
        Vindex voxel_pos = L_G(tid, voxel_pos);
        Vindex voxel_neg_mat = V_G(voxel_neg, voxel_material);
        Vindex voxel_pos_mat = V_G(voxel_pos, voxel_material);

        if (L_G(tid, removed))
            return;
        if (VM_G(voxel_neg_mat, fixed) && VM_G(voxel_pos_mat, fixed))
            return;
        if (L_G(tid, is_detached))
            return;
        VX3_Link::timeStep(ctx, tid);
    }
}

__device__ void VX3_VoxelyzeKernel::updateVoxels() {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < ctx.voxels.size()) {
        Vindex material = V_G(tid, voxel_material);
        if (VM_G(material, fixed))
            return; // fixed voxels, no need to update position
        VX3_Voxel::timeStep(*this, tid, dt, time);
    }
}

__device__ void VX3_VoxelyzeKernel::updateVoxelTemperature() {
    // updates the temperatures For Actuation!
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (enable_vary_temp and temp_period > 0 and tid < ctx.voxels.size()) {
        Vindex material = V_G(tid, voxel_material);
        if (VM_G(material, fixed))
            return; // fixed voxels, no need to update temperature

        Vfloat amplitude = V_G(tid, amplitude);
        Vfloat frequency = V_G(tid, frequency);
        Vfloat phase_offset = V_G(tid, phase_offset);

        Vfloat currentTemperature =
            temp_amplitude * amplitude *
            sin(2 * 3.1415926f * frequency * (time / temp_period + phase_offset));

        VX3_Voxel::updateTemperature(ctx, tid, currentTemperature);
    }
}

__device__ void VX3_VoxelyzeKernel::saveRecordFrame() {
    unsigned int frame = frame_num;
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Vfloat scale = 1 / rescale;

    if (tid == 0) {
        d_steps[frame] = step;
        d_time_points[frame] = time;
    }

    if (tid < ctx.voxels.size() and record_voxel) {
        size_t offset = ctx.voxels.size() * frame;
        VX3_SimulationVoxelRecord v;
        v.valid = VX3_Voxel::isSurface(ctx, tid);
        if (v.valid) {
            v.material = V_G(tid, voxel_material);
            v.local_signal = V_G(tid, local_signal);
            auto position = V_G(tid, position);
            v.x = position.x * scale;
            v.y = position.y * scale;
            v.z = position.z * scale;
            auto orientation = V_G(tid, orientation);
            v.orient_angle = orientation.angleDegrees();
            v.orient_x = orientation.x;
            v.orient_y = orientation.y;
            v.orient_z = orientation.z;
            Vec3f ppp = V_G(tid, ppp_offset), nnn = V_G(tid, nnn_offset);
            v.nnn_x = nnn.x * scale;
            v.nnn_y = nnn.y * scale;
            v.nnn_z = nnn.z * scale;
            v.ppp_x = ppp.x * scale;
            v.ppp_y = ppp.y * scale;
            v.ppp_z = ppp.z * scale;
        }
        d_voxel_record[offset + tid] = v;
    }
    if (tid < ctx.links.size() and record_link) {
        size_t offset = ctx.links.size() * frame;
        VX3_SimulationLinkRecord l;
        l.valid = not L_G(tid, is_detached);
        if (l.valid) {
            auto pos_voxel_position = V_G(L_G(tid, voxel_pos), position);
            auto neg_voxel_position = V_G(L_G(tid, voxel_neg), position);
            l.pos_x = pos_voxel_position.x;
            l.pos_y = pos_voxel_position.y;
            l.pos_z = pos_voxel_position.z;
            l.neg_x = neg_voxel_position.x;
            l.neg_y = neg_voxel_position.y;
            l.neg_z = neg_voxel_position.z;
        }
        d_link_record[offset + tid] = l;
    }
}
