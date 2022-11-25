#include "vx3_simulation_manager.h"

#include <boost/format.hpp>
#include <iostream>
#include <queue>
#include <stack>
#include <utility>

#include "vx3/vx3_voxelyze_kernel.cuh"


void VX3_SimulationManager::initSim(const VX3_Config &config, int device_index) {
    VcudaSetDevice(device_index);
    VcudaStreamCreate(&stream);
    kernel = kernel_manager.createKernelFromConfig(config, stream);
    // rescale the whole space. so history file can contain less digits.
    // ( e.g. not 0.000221, but 2.21 )
    record.rescale = 0.001;

    kernel.init(-1, record.rescale);
}

bool VX3_SimulationManager::runSim(int max_steps) {
    if (kernel.ctx.voxels.size() == 0 and kernel.ctx.links.size() == 0) {
        std::cout << COLORCODE_BOLD_RED
            "No links and no voxels. Simulation abort.\n" COLORCODE_RESET
                  << std::endl;
        return false;
    }
    for (int step = 0; step < max_steps; step++) { // Maximum Steps 1000000
        if (kernel.isStopConditionMet())
            break;
        if (not kernel.doTimeStep()) {
            std::cout << COLORCODE_BOLD_RED "Simulation diverged.\n" COLORCODE_RESET
                      << std::endl;
            return false;
        }
    }
    VcudaStreamSynchronize(stream);
    record.vox_size = kernel.vox_size;
    record.dt_frac = kernel.dt_frac;
    record.recommended_time_step = kernel.recommended_time_step;
    record.real_step_size = kernel.real_step_size;
    for (auto &voxel_material : kernel_manager.ictx.voxel_materials) {
        record.voxel_materials.emplace_back(std::make_tuple(
            voxel_material.material_id, Vfloat(voxel_material.r) / 255.,
            Vfloat(voxel_material.g) / 255., Vfloat(voxel_material.b) / 255.,
            Vfloat(voxel_material.a) / 255.));
    }

    saveResult();
    saveRecord();
    kernel_manager.freeKernel(kernel);
    VcudaStreamDestroy(stream);
    return true;
}

VX3_SimulationRecord VX3_SimulationManager::getRecord() { return std::move(record); }

VX3_SimulationResult VX3_SimulationManager::getResult() { return std::move(result); }

void VX3_SimulationManager::saveResult() {
    // insert results to h_results
    result.current_time = kernel.time;
    result.fitness_score = kernel.computeFitness();
    result.initial_center_of_mass = kernel.initial_center_of_mass;
    result.current_center_of_mass = kernel.current_center_of_mass;
    result.num_close_pairs = kernel.num_close_pairs;

    // Save voxel related states
    result.vox_size = kernel.vox_size;
    result.num_voxel = (int)kernel.ctx.voxels.size();
    result.save_position_of_all_voxels = kernel.save_position_of_all_voxels;

    std::vector<VX3_Voxel> voxels;
    std::vector<VX3_VoxelMaterial> voxel_materials;
    kernel.ctx.voxels.read(voxels, stream);
    kernel.ctx.voxel_materials.read(voxel_materials, stream);

    result.num_measured_voxel = 0;
    result.total_distance_of_all_voxels = 0.0;
    for (auto &voxel : voxels) {
        result.voxel_initial_positions.push_back(voxel.initial_position);
        result.voxel_final_positions.push_back(voxel.position);
        result.voxel_materials.push_back((int)voxel.voxel_material);
        if (voxel_materials[voxel.voxel_material].is_measured) {
            result.num_measured_voxel++;
            result.total_distance_of_all_voxels +=
                voxel.position.dist(voxel.initial_position);
        }
    }
}

void VX3_SimulationManager::saveRecord() {
    if (kernel.frame_num == 0)
        return;
    auto tmp_steps = new unsigned long[kernel.frame_num];
    auto tmp_time_points = new Vfloat[kernel.frame_num];
    auto tmp_link_record =
        new VX3_SimulationLinkRecord[kernel.ctx.links.size() * kernel.frame_num];
    auto tmp_voxel_record =
        new VX3_SimulationVoxelRecord[kernel.ctx.voxels.size() * kernel.frame_num];

    VcudaMemcpyAsync(tmp_steps, kernel.d_steps, sizeof(unsigned long) * kernel.frame_num,
                     cudaMemcpyDeviceToHost, stream);
    VcudaMemcpyAsync(tmp_time_points, kernel.d_time_points,
                     sizeof(Vfloat) * kernel.frame_num, cudaMemcpyDeviceToHost, stream);
    VcudaMemcpyAsync(tmp_link_record, kernel.d_link_record,
                     sizeof(VX3_SimulationLinkRecord) * kernel.ctx.links.size() *
                         kernel.frame_num,
                     cudaMemcpyDeviceToHost, stream);
    VcudaMemcpyAsync(tmp_voxel_record, kernel.d_voxel_record,
                     sizeof(VX3_SimulationVoxelRecord) * kernel.ctx.voxels.size() *
                         kernel.frame_num,
                     cudaMemcpyDeviceToHost, stream);
    record.steps.assign(tmp_steps, tmp_steps + kernel.frame_num);
    record.time_points.assign(tmp_time_points, tmp_time_points + kernel.frame_num);

    for (size_t f = 0; f < kernel.frame_num; f++) {
        std::vector<VX3_SimulationLinkRecord> link_record;
        size_t offset = f * kernel.ctx.links.size();
        for (size_t l = 0; l < kernel.ctx.links.size(); l++) {
            if (tmp_link_record[offset + l].valid) {
                link_record.emplace_back(tmp_link_record[offset + l]);
            }
        }
        record.link_frames.emplace_back(link_record);
    }

    for (size_t f = 0; f < kernel.frame_num; f++) {
        std::vector<VX3_SimulationVoxelRecord> voxel_record;
        size_t offset = f * kernel.ctx.voxels.size();
        for (size_t v = 0; v < kernel.ctx.voxels.size(); v++) {
            if (tmp_voxel_record[offset + v].valid) {
                voxel_record.emplace_back(tmp_voxel_record[offset + v]);
            }
        }
        record.voxel_frames.emplace_back(voxel_record);
    }

    delete[] tmp_steps;
    delete[] tmp_time_points;
    delete[] tmp_link_record;
    delete[] tmp_voxel_record;
}