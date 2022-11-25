#ifndef VX3_VOXEL_H
#define VX3_VOXEL_H

// #include "VX_Voxel.h" // for CVX_Voxel
#include "utils/vx3_cuda.cuh"
#include "utils/vx3_def.h"
#include "utils/vx3_quat3d.h"
#include "utils/vx3_soa.h"
#include "utils/vx3_vec3d.h"
#include "vx3/vx3_signal.h"

class VX3_VoxelyzeKernel;
struct VX3_InitContext;
struct VX3_Context;

class VX3_Voxel {
  public:
    VX3_Voxel() = default;

    /*****************************************************************
     * Host side methods
     *****************************************************************/
    //!< Initialize, check pre-set values, and updates all pos-set values.
    //!< This function is called by the kernel manager.
    void init(const VX3_InitContext &ictx);

    /*****************************************************************
     * Device side methods
     *****************************************************************/
    //!< Advances this voxel's state according to all forces
    //!< and moments acting on it. Large timesteps will cause
    //!< instability. Use CVoxelyze::recommendedTimeStep() to
    //!< get the recommended largest stable timestep.
    //!< @param[in] dt Timestep (in second) to advance.
    __device__ static void timeStep(VX3_VoxelyzeKernel &k, Vindex voxel, Vfloat dt,
                                    Vfloat current_time);

    //!< Specifies the temperature for this voxel. This
    //!< adds (or subtracts) the correct amount of thermal
    //!< energy to leave the voxel at ths specified
    //!< temperature, but this temperature will not be
    //!< maintaned without subsequent determines the
    //!< amount of scaling from the temperature
    __device__ static void updateTemperature(VX3_Context &ctx, Vindex voxel,
                                             Vfloat temperature);

    //!< Returns the nominal size of this voxel (LCS) accounting for any
    //!< specified temperature and external actuation. Specifically, returns
    //!< the zero-stress size of the voxel if all forces/moments were removed.
    // currently, all three axis are equal
    __device__ static Vec3f baseSize(const VX3_Context &ctx, Vindex voxel);

    //!< Returns the nominal size of this voxel in the specified axis
    //!< accounting for any specified temperature and external actuation.
    //!< Specifically, returns the zero-stress dimension of the voxel if all
    //!< forces/moments were removed.
    __device__ static Vfloat baseSize(const VX3_Context &ctx, Vindex voxel,
                                      LinkAxis axis) {
        return baseSize(ctx, voxel)[axis];
    }

    //!< Returns the deformed location of the voxel corner
    //!< in the specified corner in the global coordinate
    //!< system (GCS). Essentially cornerOffset() with the
    //!< voxel's current global position/rotation applied.
    __device__ static Vec3f cornerPosition(const VX3_Context &ctx, Vindex voxel,
                                           VoxelCorner corner);

    //!< Returns the deformed location of the voxel corner in the
    //!< specified corner in the local voxel coordinate system (LCS).
    //!< Used to draw the deformed voxel in the correct position
    //!< relative to the position().
    __device__ static Vec3f cornerOffset(const VX3_Context &ctx, Vindex voxel,
                                         VoxelCorner corner);

    //!< Returns the average nominal size of the voxel in a zero-stress (no
    //!< force) state. (X+Y+Z/3)
    __device__ static Vfloat baseSizeAverage(const VX3_Context &ctx, Vindex voxel) {
        auto base_size = baseSize(ctx, voxel);
        return base_size.mean();
    }

    //!< Returns the transverse area of this voxel with respect
    //!< to the specified axis. This would normally be called
    //!< only internally, but can be used to calculate the
    //!< correct relationship between force and stress for this
    //!< voxel if Poisson's ratio is non-zero.
    __device__ static Vfloat transverseArea(const VX3_Context &ctx, Vindex voxel,
                                            LinkAxis axis);

    //!< Returns the sum of the current strain of this voxel in
    //!< the two mutually perpendicular axis to the specified
    //!< axis. This would normally be called only internally, but
    //!< can be used to correctly calculate stress for this voxel
    //!< if Poisson's ratio is non-zero.
    __device__ static Vfloat transverseStrainSum(const VX3_Context &ctx, Vindex voxel,
                                                 LinkAxis axis);

    //!< Returns the interference (in meters) between the collision envelope
    //!< of this voxel and the floor at Z=0. >=0 numbers correspond to
    //!< interference. If the voxel is not touching the floor return a value < 0.
    __device__ static Vfloat floorPenetration(const VX3_Context &ctx, Vindex voxel);

    //!< Calculates and returns the sum of the current
    //!< forces on this voxel. This would normally only be
    //!< called internally, but can be used to query the
    //!< state of a voxel for visualization or debugging.
    __device__ static Vec3f force(const VX3_Context &ctx, Vindex voxel,
                                  Vfloat grav_acc = -9.80665);

    //!< Calculates and returns the sum of the current
    //!< moments on this voxel. This would normally only be
    //!< called internally, but can be used to query the
    //!< state of a voxel for visualization or debugging.
    __device__ static Vec3f moment(const VX3_Context &ctx, Vindex voxel);

    //!< Returns voxel strain. (LCS?)
    __device__ static Vec3f strain(const VX3_Context &ctx, Vindex voxel, bool poissons_strain);

    //!< Returns the 3D velocity of this voxel in m/s (GCS)
    __device__ static Vec3f velocity(const VX3_Context &ctx, Vindex voxel);

    //!< Returns the 3D angular velocity of this voxel in rad/s (GCS)
    __device__ static Vec3f angularVelocity(const VX3_Context &ctx, Vindex voxel);

    //!< Returns the damping multiplier for this voxel. This would normally be
    //!< called only internally for the internal damping calculations.
    __device__ static Vfloat dampingMultiplier(const VX3_Context &ctx, Vindex voxel);

    __device__ static bool isSurface(const VX3_Context &ctx, Vindex voxel);

    //!< Get a bool state flag
    __device__ static bool getBoolState(const VX3_Context &ctx, Vindex voxel,
                                        VoxFlags flag);

    /**
     * Pre-set attributes
     */
    // Global X, Y, Z index of this voxel.
    short index_x = -1, index_y = -1, index_z = -1;

    Vindex voxel_material = NULL_INDEX;
    // links in the 6 cardinal directions according to
    // LinkDirection enumeration
    Vindex links[6] = {NULL_INDEX, NULL_INDEX, NULL_INDEX,
                       NULL_INDEX, NULL_INDEX, NULL_INDEX};

    Vfloat amplitude = 0;
    Vfloat frequency = 0;
    Vfloat phase_offset = 0;

    Vec3f base_cilia_force;
    Vec3f shift_cilia_force;

    // voxel state
    // Initial position (not modified during sim)
    Vec3f initial_position;

    // The center position of this voxel in meters (GCS). This is
    // the origin of the local coordinate system (LCS).
    Vec3f position;

    /**
     * Post-set attributes
     */
    /// cornerOffset of NNN and PPP
    Vec3f nnn_offset, ppp_offset;

    // current linear momentum (kg*m/s) (GCS)
    Vec3f linear_momentum;

    //!< Orientation of this voxel in quaternion form (GCS). This
    //!< orientation defines the relative orientation of the local coordinate
    //!< system (LCS). The unit quaternion represents the original orientation
    //!< of this voxel.
    Quat3f orientation;

    // current angular momentum (kg*m^2/s) (GCS)
    Vec3f angular_momentum;

    // single int to store many boolean state values as
    // bit flags according to
    VoxState bool_states = 0;

    // 0 is no expansion
    Vfloat temperature = 0;

    // cached poissons strain
    Vec3f poissons_strain;

    // remember the duration of the last timestep of this voxel
    Vfloat previous_dt = 0;

    Vec3f last_col_watch_position;

    // true if the voxel is on main body, false if it fell on
    // the ground.
    bool is_detached = false;

    Vec3f contact_force;

    Vec3f cilia_force;

    bool enable_attach = false;

    VX3_Signal signal;
    Vfloat local_signal = 0.0;
    Vfloat local_signal_dt = 0.0;
    Vfloat packmaker_next_pulse = 0.0;
    Vfloat inactive_until = 0.0;

private:
    /*****************************************************************
     * Device side methods
     *****************************************************************/
    //!< modifies pTotalForce to include the
    //!< object's interaction with a floor. This
    //!< should be calculated as the last step of
    //!< sumForce so that pTotalForce is complete.
    __device__ static void floorForce(VX3_Context &ctx, Vindex voxel, float dt,
                                      Vec3f &total_force);

    //!< Set/Clear a bool state flag
    __device__ static void setBoolState(VX3_Context &ctx, Vindex voxel, VoxFlags flag,
                                        bool active);
};

REFL_AUTO(type(VX3_Voxel), field(voxel_material), field(links), field(index_x),
          field(index_y), field(index_z), field(position), field(nnn_offset),
          field(ppp_offset), field(linear_momentum), field(orientation),
          field(angular_momentum), field(bool_states), field(temperature),
          field(poissons_strain), field(previous_dt),
          field(last_col_watch_position), field(amplitude), field(frequency),
          field(phase_offset), field(is_detached), field(contact_force),
          field(base_cilia_force), field(shift_cilia_force), field(cilia_force),
          field(enable_attach), field(signal), field(local_signal),
          field(local_signal_dt), field(packmaker_next_pulse), field(inactive_until))

#endif // VX3_VOXEL_H
