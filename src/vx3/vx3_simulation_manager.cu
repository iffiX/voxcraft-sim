#include "vx3_simulation_manager.h"

#include <iostream>
#include <queue>
#include <stack>
#include <thread>
#include <fmt/format.h>
#include "utils/vx3_conf.h"
#include "vx3/vx3_voxelyze_kernel.cuh"

using namespace std;
using namespace fmt;
using namespace boost;

void VX3_SimulationManager::addSim(const VX3_Config &config) {
#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}) begin adding sim {}\n", device, batch, sims.size());
#endif
    sims.emplace_back();
    auto &sim = sims.back();
    sim.kernel = sim.kernel_manager.createKernelFromConfig(config, stream);
    // rescale the whole space. so history file can contain less digits.
    // ( e.g. not 0.000221, but 2.21 )
    sim.record.rescale = VX3_RECORD_RESCALE;
#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}) end adding sim {}\n", device, batch, sims.size());
#endif
}

vector<bool> VX3_SimulationManager::runSims(int max_steps) {
    if (sims.empty())
        return {};

    size_t sim_index = 0;
    vector<bool> result(sims.size(), true);
    for (auto &sim : sims) {
        if (sim.kernel.ctx.voxels.size() == 0 and sim.kernel.ctx.links.size() == 0) {
            cout << format("(Device {}, Batch {})", device, batch)
                 << COLORCODE_BOLD_RED << "No links and no voxels. Simulation "
                 << sim_index << " abort. " << COLORCODE_RESET << endl;
            sim.is_finished = true;
            result[sim_index] = false;
        }
        sim_index++;
    }

    vector<VX3_VoxelyzeKernel *> kernels;
    vector<size_t> sim_indices;
    sim_index = 0;
    for (auto &sim : sims) {
        if (not sim.is_finished) {
            kernels.push_back(&sim.kernel);
            sim_indices.push_back(sim_index);
        }
        sim_index++;
    }

#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}) begin init sims\n", device, batch);
#endif
    exec.init(kernels, stream, -1, VX3_RECORD_RESCALE);
#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}) end init sims\n", device, batch);
#endif

    for (int step = 0; step < max_steps; step++) { // Maximum Steps 1000000
        vector<size_t> kernel_indices;
        for (size_t i = 0; i < kernels.size(); i++) {
            auto &sim = sims[sim_indices[i]];
            if (not sim.is_finished and sim.kernel.isStopConditionMet()) {
                sim.is_finished = true;
            }
            if (not sim.is_finished) {
                kernel_indices.push_back(i);
            }
        }

        if (kernel_indices.empty())
            break;

        auto step_result = exec.doTimeStep(kernel_indices);
        for (size_t i = 0; i < kernel_indices.size(); i++) {
            if (not step_result[i]) {
                cout << format("(Device {}, Batch {})", device, batch)
                     << COLORCODE_BOLD_RED "Simulation " << kernel_indices[i]
                     << " diverged.\n" COLORCODE_RESET << endl;
                sims[kernel_indices[i]].is_finished = true;
                result[kernel_indices[i]] = false;
            }
        }
#ifdef DEBUG_SIMULATION_MANAGER
        print("(Device {}, Batch {}) step {} finished\n", device, batch, step);
#endif
    }
    VcudaStreamSynchronize(stream);
    vector<thread> save_workers;
#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}) starting save workers\n", device, batch);
#endif
    for (size_t i = 0; i < sims.size(); i++) {
        save_workers.emplace_back(VX3_SimulationManager::finishSim, std::ref(sims[i]), stream,
                                  result[i], device, batch, i);
    }
    for (auto &worker : save_workers)
        worker.join();
#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}) saving finished\n", device, batch);
#endif

    exec.free();

    VcudaStreamDestroy(stream);
#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}) cleaning up finished\n", device, batch);
#endif
    return result;
}

void VX3_SimulationManager::finishSim(Simulation &sim, cudaStream_t stream,
                                      bool has_no_exception,
                                      int device, int batch, int sim_index) {
    // For sub threads, we also need to set device,
    // otherwise calls will go to GPU 0
    VcudaSetDevice(device);
#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}, Sim {}) worker started\n", device, batch, sim_index);
#endif
    if (has_no_exception) {
        sim.record.vox_size = sim.kernel.vox_size;
        sim.record.dt_frac = sim.kernel.dt_frac;
        sim.record.recommended_time_step = sim.kernel.recommended_time_step;
        sim.record.real_step_size = sim.kernel.real_step_size;
        for (auto &voxel_material : sim.kernel_manager.ictx.voxel_materials) {
            sim.record.voxel_materials.emplace_back(
                voxel_material.material_id, Vfloat(voxel_material.r) / 255.,
                Vfloat(voxel_material.g) / 255., Vfloat(voxel_material.b) / 255.,
                Vfloat(voxel_material.a) / 255.);
        }
#ifdef DEBUG_SIMULATION_MANAGER
        print("(Device {}, Batch {}, Sim {}) begin saving\n", device, batch, sim_index);
#endif
        saveResult(sim, stream);
#ifdef DEBUG_SIMULATION_MANAGER
        print("(Device {}, Batch {}, Sim {}) result saved\n", device, batch, sim_index);
#endif
        saveRecord(sim, stream);
#ifdef DEBUG_SIMULATION_MANAGER
        print("(Device {}, Batch {}, Sim {}) record saved\n", device, batch, sim_index);
#endif
    }
    sim.kernel_manager.freeKernel(sim.kernel, stream);

#ifdef DEBUG_SIMULATION_MANAGER
    print("(Device {}, Batch {}, Sim {}) kernel freed\n", device, batch, sim_index);
#endif
}

void VX3_SimulationManager::saveResult(Simulation &sim, cudaStream_t stream) {
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

    vector<VX3_Voxel> voxels;
    vector<VX3_VoxelMaterial> voxel_materials;
    sim.kernel.ctx.voxels.read(voxels, stream);
    sim.kernel.ctx.voxel_materials.read(voxel_materials, stream);
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

void VX3_SimulationManager::saveRecord(Simulation &sim, cudaStream_t stream) {
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
                     stream);
    VcudaMemcpyAsync(tmp_time_points, sim.kernel.d_time_points,
                     sizeof(Vfloat) * sim.kernel.frame_num, cudaMemcpyDeviceToHost,
                     stream);
    VcudaMemcpyAsync(tmp_link_record, sim.kernel.d_link_record,
                     sizeof(VX3_SimulationLinkRecord) * sim.kernel.ctx.links.size() *
                         sim.kernel.frame_num,
                     cudaMemcpyDeviceToHost, stream);
    VcudaMemcpyAsync(tmp_voxel_record, sim.kernel.d_voxel_record,
                     sizeof(VX3_SimulationVoxelRecord) * sim.kernel.ctx.voxels.size() *
                         sim.kernel.frame_num,
                     cudaMemcpyDeviceToHost, stream);
    // Make sure all device side data are transferred
    VcudaStreamSynchronize(stream);

    sim.record.steps.assign(tmp_steps, tmp_steps + sim.kernel.frame_num);
    sim.record.time_points.assign(tmp_time_points,
                                  tmp_time_points + sim.kernel.frame_num);

    for (size_t f = 0; f < sim.kernel.frame_num; f++) {
        vector<VX3_SimulationLinkRecord> link_record;
        size_t offset = f * sim.kernel.ctx.links.size();
        for (size_t l = 0; l < sim.kernel.ctx.links.size(); l++) {
            if (tmp_link_record[offset + l].valid) {
                link_record.emplace_back(tmp_link_record[offset + l]);
            }
        }
        sim.record.link_frames.emplace_back(link_record);
    }

    for (size_t f = 0; f < sim.kernel.frame_num; f++) {
        vector<VX3_SimulationVoxelRecord> voxel_record;
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