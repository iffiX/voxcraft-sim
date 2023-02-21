#ifndef VX3_SIMULATION_MANAGER
#define VX3_SIMULATION_MANAGER
#include <boost/filesystem.hpp>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include "vx3/vx3_simulation_record.h"
#include "vx3/vx3_voxelyze_kernel.cuh"
#include "vx3/vx3_voxelyze_kernel_manager.cuh"
#include "vxa/vx3_config.h"

class VX3_SimulationManager {
  public:
    struct Simulation {
        bool is_finished = false;
        VX3_VoxelyzeKernelManager kernel_manager;
        VX3_VoxelyzeKernel kernel;
        VX3_SimulationRecord record;
        VX3_SimulationResult result;
    };

  public:
    int device;
    int batch;
    cudaStream_t stream;
    VX3_VoxelyzeKernelBatchExecutor exec;
    std::vector<Simulation> sims;

    VX3_SimulationManager(int device, int batch)
        : device(device), batch(batch) {
        VcudaSetDevice(device);
        VcudaStreamCreate(&stream);
    };
    void addSim(const VX3_Config &config);
    std::vector<bool> runSims(int max_steps = 1000000);

  private:
    static void finishSim(Simulation &sim, cudaStream_t stream, bool has_no_exception,
                          int device, int batch, int sim_index);
    static void saveResult(Simulation &sim, cudaStream_t stream);
    static void saveRecord(Simulation &sim, cudaStream_t stream);
};

#endif // VX3_SIMULATION_MANAGER
