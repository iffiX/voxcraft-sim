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
        cudaStream_t stream;
        bool is_finished = false;
        VX3_VoxelyzeKernelManager kernel_manager;
        VX3_VoxelyzeKernel kernel;
        VX3_SimulationRecord record;
        VX3_SimulationResult result;
    };
  public:
    int device_index;
    std::vector<Simulation> sims;

    VX3_SimulationManager(int device_index = 0) : device_index(device_index) {};
    void addSim(const VX3_Config &config);
    std::vector<bool> runSims(int max_steps = 1000000);

  private:
    void saveResult(int sim_index);
    void saveRecord(int sim_index);
};

#endif // VX3_SIMULATION_MANAGER
