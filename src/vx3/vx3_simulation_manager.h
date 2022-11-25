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
    VX3_SimulationManager() = default;
    void initSim(const VX3_Config &config, int device_index);
    bool runSim(int max_steps = 1000000);
    VX3_SimulationRecord getRecord();
    VX3_SimulationResult getResult();

  private:
    cudaStream_t stream;
    VX3_VoxelyzeKernelManager kernel_manager;
    VX3_VoxelyzeKernel kernel;
    VX3_SimulationRecord record;
    VX3_SimulationResult result;
    void saveResult();
    void saveRecord();
};

#endif // VX3_SIMULATION_MANAGER
