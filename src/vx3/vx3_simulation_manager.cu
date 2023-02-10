#include "vx3_simulation_manager.h"

#include <boost/format.hpp>
#include <iostream>
#include <queue>
#include <stack>
#include <utility>

#include "vx3/vx3_voxelyze_kernel.cuh"

void VX3_SimulationManager::addSim(const VX3_Config &config) {
    VcudaSetDevice(device_index);
    sims.emplace_back(Simulation());
    auto &sim = sims.back();
    VcudaStreamCreate(&sim.stream);
    sim.kernel = sim.kernel_manager.createKernelFromConfig(config, sim.stream);
    // rescale the whole space. so history file can contain less digits.
    // ( e.g. not 0.000221, but 2.21 )
    sim.record.rescale = 0.001;

    sim.kernel.init(-1, sim.record.rescale);
}

std::vector<bool> VX3_SimulationManager::runSims(int max_steps) {
    if (sims.empty())
        return std::vector<bool>();
    else if (sims.size() == 1) {
        // Use update function for non-batched kernel to reduce
        // stream synchronization cost in doTimeStep()
        auto &kernel = sims[0].kernel;
        auto &kernel_manager = sims[0].kernel_manager;
        auto &stream = sims[0].stream;
        auto &record = sims[0].record;

        if (kernel.ctx.voxels.size() == 0 and kernel.ctx.links.size() == 0) {
            std::cout << COLORCODE_BOLD_RED
                         "No links and no voxels. Simulation 0 abort.\n" COLORCODE_RESET
                      << std::endl;
            return {false};
        }
        for (int step = 0; step < max_steps; step++) { // Maximum Steps 1000000
            if (kernel.isStopConditionMet())
                break;
            if (not kernel.doTimeStep()) {
                std::cout << COLORCODE_BOLD_RED "Simulation 0 diverged.\n" COLORCODE_RESET
                          << std::endl;
                return {false};
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

        saveResult(0);
        saveRecord(0);
        kernel_manager.freeKernel(kernel);
        VcudaStreamDestroy(stream);
        return {true};
    }
    else {
        int sim_index = 0;
        std::vector<bool> result(sims.size(), true);
        for (auto &sim : sims) {
            if (sim.kernel.ctx.voxels.size() == 0 and sim.kernel.ctx.links.size() == 0) {
                std::cout << COLORCODE_BOLD_RED << "No links and no voxels. Simulation "
                          << sim_index << " abort.\n"
                          << COLORCODE_RESET << std::endl;
                sim.is_finished = true;
                result[sim_index] = false;
            }
            sim_index++;
        }

        auto &stream = sims[0].stream;
        for (int step = 0; step < max_steps; step++) { // Maximum Steps 1000000
            std::vector<VX3_VoxelyzeKernel *> kernels;
            std::vector<int> kernel_index;

            sim_index = 0;
            for (auto &sim : sims) {
                if (not sim.is_finished and sim.kernel.isStopConditionMet()) {
                    sim.is_finished = true;
                }
                if (not sim.is_finished) {
                    kernels.push_back(&sim.kernel);
                    kernel_index.push_back(sim_index);
                }
                sim_index++;
            }
            auto step_result = VX3_VoxelyzeKernel::doTimeStepBatch(kernels, stream);
            for (size_t i = 0; i < kernels.size(); i++) {
                if (not step_result[i]) {
                    std::cout << COLORCODE_BOLD_RED "Simulation " << kernel_index[i]
                              << " diverged.\n" COLORCODE_RESET << std::endl;
                    sims[kernel_index[i]].is_finished = true;
                    result[kernel_index[i]] = false;
                }
            }
        }

        VcudaStreamSynchronize(stream);
        sim_index = 0;
        for (auto &sim : sims) {
            sim.record.vox_size = sim.kernel.vox_size;
            sim.record.dt_frac = sim.kernel.dt_frac;
            sim.record.recommended_time_step = sim.kernel.recommended_time_step;
            sim.record.real_step_size = sim.kernel.real_step_size;
            for (auto &voxel_material: sim.kernel_manager.ictx.voxel_materials) {
                sim.record.voxel_materials.emplace_back(std::make_tuple(
                        voxel_material.material_id, Vfloat(voxel_material.r) / 255.,
                        Vfloat(voxel_material.g) / 255., Vfloat(voxel_material.b) / 255.,
                        Vfloat(voxel_material.a) / 255.));
            }

            saveResult(sim_index);
            saveRecord(sim_index);
            sim.kernel_manager.freeKernel(sim.kernel);
            VcudaStreamDestroy(sim.stream);
            sim_index++;
        }
        return result;
    }
}

void VX3_SimulationManager::saveResult(int sim_index) {
    auto &sim = sims[sim_index];

    // insert results to h_results
    sim.result.current_time = sim.kernel.time;
    sim.result.fitness_score = sim.kernel.computeFitness();
    sim.result.initial_center_of_mass = sim.kernel.initial_center_of_mass;
    sim.result.current_center_of_mass = sim.kernel.current_center_of_mass;
    sim.result.num_close_pairs = sim.kernel.num_close_pairs;

    // Save voxel related states
    sim.result.vox_size = sim.kernel.vox_size;
    sim.result.num_voxel = (int)sim.kernel.ctx.voxels.size();
    sim.result.save_position_of_all_voxels = sim.kernel.save_position_of_all_voxels;

    std::vector<VX3_Voxel> voxels;
    std::vector<VX3_VoxelMaterial> voxel_materials;
    sim.kernel.ctx.voxels.read(voxels, sim.stream);
    sim.kernel.ctx.voxel_materials.read(voxel_materials, sim.stream);

    sim.result.num_measured_voxel = 0;
    sim.result.total_distance_of_all_voxels = 0.0;
    for (auto &voxel : voxels) {
        sim.result.voxel_initial_positions.push_back(voxel.initial_position);
        sim.result.voxel_final_positions.push_back(voxel.position);
        sim.result.voxel_materials.push_back((int)voxel.voxel_material);
        if (voxel_materials[voxel.voxel_material].is_measured) {
            sim.result.num_measured_voxel++;
            sim.result.total_distance_of_all_voxels +=
                voxel.position.dist(voxel.initial_position);
        }
    }
}

void VX3_SimulationManager::saveRecord(int sim_index) {
    auto &sim = sims[sim_index];

    if (sim.kernel.frame_num == 0)
        return;
    auto tmp_steps = new unsigned long[sim.kernel.frame_num];
    auto tmp_time_points = new Vfloat[sim.kernel.frame_num];
    auto tmp_link_record =
        new VX3_SimulationLinkRecord[sim.kernel.ctx.links.size() * sim.kernel.frame_num];
    auto tmp_voxel_record = new VX3_SimulationVoxelRecord[sim.kernel.ctx.voxels.size() *
                                                          sim.kernel.frame_num];

    VcudaMemcpyAsync(tmp_steps, sim.kernel.d_steps,
                     sizeof(unsigned long) * sim.kernel.frame_num, cudaMemcpyDeviceToHost,
                     sim.stream);
    VcudaMemcpyAsync(tmp_time_points, sim.kernel.d_time_points,
                     sizeof(Vfloat) * sim.kernel.frame_num, cudaMemcpyDeviceToHost,
                     sim.stream);
    VcudaMemcpyAsync(tmp_link_record, sim.kernel.d_link_record,
                     sizeof(VX3_SimulationLinkRecord) * sim.kernel.ctx.links.size() *
                         sim.kernel.frame_num,
                     cudaMemcpyDeviceToHost, sim.stream);
    VcudaMemcpyAsync(tmp_voxel_record, sim.kernel.d_voxel_record,
                     sizeof(VX3_SimulationVoxelRecord) * sim.kernel.ctx.voxels.size() *
                         sim.kernel.frame_num,
                     cudaMemcpyDeviceToHost, sim.stream);
    sim.record.steps.assign(tmp_steps, tmp_steps + sim.kernel.frame_num);
    sim.record.time_points.assign(tmp_time_points, tmp_time_points + sim.kernel.frame_num);

    for (size_t f = 0; f < sim.kernel.frame_num; f++) {
        std::vector<VX3_SimulationLinkRecord> link_record;
        size_t offset = f * sim.kernel.ctx.links.size();
        for (size_t l = 0; l < sim.kernel.ctx.links.size(); l++) {
            if (tmp_link_record[offset + l].valid) {
                link_record.emplace_back(tmp_link_record[offset + l]);
            }
        }
        sim.record.link_frames.emplace_back(link_record);
    }

    for (size_t f = 0; f < sim.kernel.frame_num; f++) {
        std::vector<VX3_SimulationVoxelRecord> voxel_record;
        size_t offset = f * sim.kernel.ctx.voxels.size();
        for (size_t v = 0; v < sim.kernel.ctx.voxels.size(); v++) {
            if (tmp_voxel_record[offset + v].valid) {
                voxel_record.emplace_back(tmp_voxel_record[offset + v]);
            }
        }
        sim.record.voxel_frames.emplace_back(voxel_record);
    }

    delete[] tmp_steps;
    delete[] tmp_time_points;
    delete[] tmp_link_record;
    delete[] tmp_voxel_record;
}