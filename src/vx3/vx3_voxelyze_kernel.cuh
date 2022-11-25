#ifndef VX3_VOXELYZE_KERNEL_H
#define VX3_VOXELYZE_KERNEL_H
#include <vector>
#include <thrust/device_vector.h>
#include "utils/vx3_cuda.cuh"
#include "utils/vx3_force_field.h"
#include "vx3/vx3_link.h"
#include "vx3/vx3_voxel.h"
#include "vx3/vx3_link_material.h"
#include "vx3/vx3_voxel_material.h"
#include "vx3/vx3_context.h"
#include "vx3/vx3_simulation_record.h"

struct VX3_VoxelyzeKernel {
    /**
     * Note: memory management is done by VX3_VoxelyzeKernelManager
     */
    VX3_VoxelyzeKernel() = default;
    VX3_VoxelyzeKernel(const cudaStream_t &stream) : stream(stream) {};
    Vfloat recommendedTimeStep() const;
    bool isStopConditionMet() const;
    bool isAnyLinkDiverged() const;

    Vfloat computeFitness() const;
    Vec3f computeCurrentCenterOfMass() const;
    std::pair<int, Vfloat> computeTargetCloseness() const;

    void init(Vfloat dt = -1.0f, Vfloat rescale = 0.001);
    bool doTimeStep(int dt_update_interval = 10, int divergence_check_interval = 100,
                    bool save_frame = true);
    void adjustRecordFrameStorage(size_t required_size);
    void updateMetrics();

    // update sub-routines used by doTimeStep
    __device__ void updateLinks();
    __device__ void updateVoxels();
    __device__ void updateVoxelTemperature();
    __device__ void saveRecordFrame();

    /* data */
    // Do not access this attribute on the device side!
    cudaStream_t stream;

    // TODO: find a way to move host-side math expressions out
    /**
     * Pre-set attributes (kernel read only)
     */
    VX3_Context ctx;

    Vfloat vox_size = 0;            // lattice size

    // In VXA.Simulator.Integration
    Vfloat dt_frac = 0;

    // In VXA.Simulator.StopCondition
    VX3_MathTreeToken stop_condition_formula[MAX_EXPRESSION_TOKENS];

    // In VXA.Simulator.RecordHistory
    int record_step_size = 0;
    bool record_link = false;
    bool record_voxel = false;

    // In VXA.Simulator.AttachDetach (collision mechanism not implemented yet)
    bool enable_attach = false;
    bool enable_detach = false;
    bool enable_collision = false;

    //// Collision constants
    Vfloat bounding_radius; //(in voxel units) radius to collide a voxel at
    Vfloat watch_distance;  //(in voxel units) Distance between voxels (not including 2*boundingRadius for each voxel) to watch for collisions from.

    //// Safety guard during the creation of new link
    int safety_guard = 500;
    VX3_MathTreeToken attach_conditions[5][MAX_EXPRESSION_TOKENS];

    // In VXA.Simulator.ForceField
    VX3_ForceField<MAX_EXPRESSION_TOKENS> force_field;

    // In VXA.Simulator
    VX3_MathTreeToken fitness_function[MAX_EXPRESSION_TOKENS];
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

    // Some other preset attributes
    Vindex* target_indices = nullptr;
    size_t target_num = 0;

    /**
     * Post-set attributes (kernel read only)
     */
    Vfloat recommended_time_step = 0;
    Vfloat dt = 0;
    Vfloat rescale = 1;
    bool is_dt_dynamic = false;
    int real_step_size = 0;

    Vfloat time = 0.0f; // current time of the simulation in seconds
    unsigned long step = 0;

    // All below Metrics
    Vec3f current_center_of_mass;
    Vec3f initial_center_of_mass;

    int collision_count = 0;
    int num_close_pairs = 0;
    Vfloat target_closeness = 0;

    //Calculate Angle by
    //A---B----C
    //A: current_center_of_mass_history[0]
    //B: current_center_of_mass_history[1]
    //C: current_center_of_mass
    Vec3f current_center_of_mass_history[2];
    int angle_sample_times = 0;
    Vfloat recent_angle = 0;

    // For frame recording
    unsigned int frame_storage_size = 0;
    unsigned int frame_num = 0;

    // This special memory region is a pinned host memory used to
    // perform asynchronous copy of the reduced results
    void *h_reduce_output = nullptr;

    /** Device side states and storage (kernel RW) **/
    // These special memory regions are used for reducing results
    // size = max(voxel_num, link_num) * sizeof(Vfloat) * 4
    void *d_reduce1 = nullptr, *d_reduce2 = nullptr;

    unsigned long *d_steps = nullptr;
    Vfloat *d_time_points = nullptr;
    VX3_SimulationLinkRecord *d_link_record = nullptr;
    VX3_SimulationVoxelRecord *d_voxel_record = nullptr;
};

#endif // VX3_VOXELYZE_KERNEL_H
