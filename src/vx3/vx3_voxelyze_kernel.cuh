#ifndef VX3_VOXELYZE_KERNEL_H
#define VX3_VOXELYZE_KERNEL_H

#include "utils/vx3_cuda.cuh"
#include "utils/vx3_force_field.h"
#include "vx3/vx3_context.h"
#include "vx3/vx3_link.h"
#include "vx3/vx3_link_material.h"
#include "vx3/vx3_simulation_record.h"
#include "vx3/vx3_voxel.h"
#include "vx3/vx3_voxel_material.h"
#include <vector>

struct __align__(8) VX3_VoxelyzeKernel {
    /**
     * Note: memory management is done by VX3_VoxelyzeKernelManager
     */
    bool isStopConditionMet() const;

    bool isResultStartConditionMet() const;

    bool isResultEndConditionMet() const;

    Vfloat computeFitness(const Vec3f &start_center_of_mass,
                          const Vec3f &end_center_of_mass) const;

    void adjustRecordFrameStorage(size_t required_size, cudaStream_t stream);

    // update sub-routines used by doTimeStep
    __device__ void updateLinks(Vindex local_id);

    __device__ void updateVoxels(Vindex local_id);

    __device__ void updateVoxelTemperature(Vindex local_id);

    __device__ void saveRecordFrame(Vindex tid);

    /* data */
    /**
     * Runtime data (set by host, kernel read only)
     */
    // Data structure that holds info for voxels, links, materials
    VX3_Context ctx;

    // In VXA.VXC.Lattice.Lattice_Dim
    Vfloat vox_size = 0; // lattice size

    // In VXA.Simulator.Integration
    Vfloat dt_frac = 0;

    // In VXA.Simulator.Condition
    VX3_MathTreeTokens stop_condition;
    VX3_MathTreeTokens result_start_condition;
    VX3_MathTreeTokens result_end_condition;

    // In VXA.Simulator.RecordHistory
    int record_step_size = 0;
    bool record_link = false;
    bool record_voxel = false;

    // In VXA.Simulator.AttachDetach (collision mechanism not implemented yet)
    bool enable_attach = false;
    bool enable_detach = false;
    bool enable_collision = false;

    //// Collision constants
    ////// (in voxel units) radius to collide a voxel at
    Vfloat bounding_radius;
    ////// (in voxel units) Distance between voxels (not including
    ////// 2*boundingRadius for each voxel) to watch for collisions from.
    Vfloat watch_distance;

    //// Safety guard during the creation of new link
    int safety_guard = 500;
    VX3_MathTreeTokens attach_conditions[5];

    // In VXA.Simulator.ForceField
    VX3_ForceField force_field;

    // In VXA.Simulator
    VX3_MathTreeTokens fitness_function;
    Vfloat max_dist_in_voxel_lengths_to_count_as_pair;
    bool save_position_of_all_voxels = false;
    bool enable_cilia = false;
    bool enable_signals = false;

    // In VXA.Environment.Gravity
    bool grav_enabled = false;
    bool floor_enabled = false;
    Vfloat grav_acc = 0;

    // In VXA.Environment.Thermal
    bool enable_vary_temp = false;
    Vfloat temp_amplitude = 0;
    Vfloat temp_period = 0;

    // Variables for target voxel tracing
    Vindex *d_target_indices = nullptr;
    size_t target_num = 0;

    // Variables for recording
    Vfloat rescale = 1;

    /**
     * Runtime states
     * 1. 2 copies maintained separately by device and host (host, dev)
     * 2. Maintained by host, copy to device, once on init (host->dev, once)
     * 3. Maintained by host, copy to device, multiple times during run (host->dev, multi)
     */
    // (host->dev, multi)
    Vfloat recommended_time_step = 0;
    // (host, dev)
    Vfloat dt = 0;
    // (host, dev, once)
    int real_step_size = 0;
    // (host, dev)
    Vfloat time = 0.0f; // current time of the simulation in seconds
    // (host, dev)
    Vsize step = 0;
    // (host, dev)
    Vsize frame_num = 0;

    // Variables for recording
    // (host->dev, multi)
    unsigned long *d_steps = nullptr;
    Vfloat *d_time_points = nullptr;
    VX3_SimulationLinkRecord *d_link_record = nullptr;
    VX3_SimulationVoxelRecord *d_voxel_record = nullptr;

    /**
     * Metrics (set by host, not used by kernel)
     */
    Vec3f current_center_of_mass;
    Vec3f initial_center_of_mass;

    int collision_count = 0;
    int num_close_pairs = 0;
    Vfloat target_closeness = 0;

    // Calculate Angle by
    // A---B---C
    // A: current_center_of_mass_history[0]
    // B: current_center_of_mass_history[1]
    // C: current_center_of_mass
    Vec3f current_center_of_mass_history[2];
    int angle_sample_times = 0;
    Vfloat recent_angle = 0;

    // For frame recording
    Vsize frame_storage_size = 0;
};

struct VX3_VoxelyzeKernelBatchExecutor {
    void init(const std::vector<VX3_VoxelyzeKernel *> &kernels, cudaStream_t stream,
              Vfloat dt = -1.0f, Vfloat rescale = 0.001);

    void initThreadRunInfo();

    void updateKernelIsRunning(const std::vector<size_t> &kernel_indices,
                               bool init = false);

    void free();

    const VX3_VoxelyzeKernel &getKernel(size_t kernel_index);

    std::vector<bool> doTimeStep(const std::vector<size_t> &kernel_indices,
                                 int dt_update_interval = 10,
                                 int divergence_check_interval = 100,
                                 bool save_frame = true);

    std::vector<Vfloat> recommendedTimeStep() const;

    std::vector<bool> isAnyLinkDiverged() const;

    std::vector<Vec3f> computeCurrentCenterOfMass() const;

    std::vector<std::pair<int, Vfloat>> computeTargetCloseness() const;

    void updateMetrics() const;

    cudaStream_t stream;

    bool is_dt_dynamic = false;

    // Maintain a dedicated step counter, so it will not be affected
    // by stopping a few kernels.
    Vsize step = 0;

    // Kernels
    Vsize kernel_num = 0;
    VX3_VoxelyzeKernel *h_kernels = nullptr;
    VX3_VoxelyzeKernel *d_kernels = nullptr;

    // used for runFunction, runFunctionAndReduce
    Vsize total_threads = 0;
    std::vector<Vsize> kernel_len;
    std::vector<Vsize> kernel_relative_indices;

    // These special memory regions are used for updating kernel states
    void *h_kernel_sync = nullptr;
    void *d_kernel_sync = nullptr;

    // These special memory regions are used for storing thread location
    // info during kernel execution
    Vsize *h_kernel_prefix_sums = nullptr;
    bool *h_kernel_is_running = nullptr;
    Vsize *d_kernel_prefix_sums = nullptr;
    bool *d_kernel_is_running = nullptr;

    // These special memory regions are used for reducing results
    void *h_reduce_output = nullptr;
    Vsize *h_sizes = nullptr;
    void *d_reduce1 = nullptr, *d_reduce2 = nullptr;
    Vsize *d_sizes = nullptr;
};

#endif // VX3_VOXELYZE_KERNEL_H
