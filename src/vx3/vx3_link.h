#ifndef VX3_LINK_H
#define VX3_LINK_H

#include "utils/vx3_cuda.cuh"
#include "utils/vx3_def.h"
#include "utils/vx3_quat3d.h"
#include "utils/vx3_soa.h"
#include "utils/vx3_vec3d.h"

class VX3_VoxelyzeKernel;
class VX3_Voxel;
class VX3_Link;
struct VX3_InitContext;
struct VX3_Context;

class VX3_Link {
  public:
    VX3_Link() = default;

    /*****************************************************************
     * Host side methods
     *****************************************************************/
    //!< Initialize, check pre-set values, and updates all pos-set values.
    //!< This function is called by the kernel manager.
    void init(const VX3_InitContext &ictx);

    /*****************************************************************
     * Device side methods
     *****************************************************************/
    //!< Called every timestep to calculate the forces and
    //!< moments acting between the two constituent voxels in
    //!< their current relative positions and orientations.
    __device__ static void timeStep(VX3_Context &ctx, Vindex link);

    //!< Returns true if the stress on this bond has ever
    //!< exceeded its yield stress
    __device__ static bool isYielded(const VX3_Context &ctx, Vindex link);

    //!< Returns true if the stress on this bond has ever
    //!< exceeded its failure stress
    __device__ static bool isFailed(const VX3_Context &ctx, Vindex link);

    //!< Returns the current calculated axial strain of the half of
    //!< the link contained in the specified voxel. @param[in]
    //!< positive_end Specifies which voxel information is desired
    //!< about.
    __device__ static Vfloat axialStrain(const VX3_Context &ctx, Vindex link,
                                         bool positive_end);

    //!< Calculates and return the strain energy of this
    //!< link according to current forces and moments.
    //!< (units: Joules, or Kg m^2 / s^2)
    __device__ static Vfloat strainEnergy(const VX3_Context &ctx, Vindex link);

    //!< Calculates and returns the current linear axial
    //!< stiffness of this link at it's current strain.
    __device__ static Vfloat axialStiffness(const VX3_Context &ctx, Vindex link);

    //!< Get bool states
    __device__ static bool getBoolState(const VX3_Context &ctx, Vindex link, LinkFlags flag);

    //!< Transforms a VX3_Vec3D in the original orientation of the bond to that
    //!< as if the bond was in +X direction.
    template <typename T>
    __host__ __device__ static VX3_Vec3D<T> toAxisX(LinkAxis axis, const VX3_Vec3D<T> &v) {
        switch (axis) {
        case Y_AXIS:
            return VX3_Vec3D<T>(v.y, -v.x, v.z);
        case Z_AXIS:
            return VX3_Vec3D<T>(v.z, v.y, -v.x);
        default:
            return v;
        }
    }

    //!< Transforms a VX3_Quat3D in the original orientation of the bond to that
    //!< as if the bond was in +X direction.
    template <typename T>
    __host__ __device__ static VX3_Quat3D<T> toAxisX(LinkAxis axis, const VX3_Quat3D<T> &q) {
        switch (axis) {
        case Y_AXIS:
            return VX3_Quat3D<T>(q.w, q.y, -q.x, q.z);
        case Z_AXIS:
            return VX3_Quat3D<T>(q.w, q.z, q.y, -q.x);
        default:
            return q;
        }
    }

    //!< Reverse the transformation on Vec3D/Quat3D.
    //!< This is an in-place transformation.
    template <typename T>
    __host__ __device__ static VX3_Vec3D<T> toAxisOriginal(LinkAxis axis, const VX3_Vec3D<T> &v) {
        switch (axis) {
        case Y_AXIS:
            return VX3_Vec3D<T>(-v.y, v.x, v.z);
        case Z_AXIS:
            return VX3_Vec3D<T>(-v.z, v.y, v.x);
        default:
            return v;
        }
    }

    /**
     * Pre-set attributes
     */
    Vindex voxel_neg = NULL_INDEX, voxel_pos = NULL_INDEX;
    Vindex link_material = NULL_INDEX;
    LinkAxis axis = X_AXIS;

    /**
     * Post-set attributes
     */
    // single int to store many boolean state values as
    // bit flags according to
    LinkState bool_states = 0;

    Vec3f force_neg, force_pos;
    Vec3f moment_neg, moment_pos;

    // axial strain
    Vfloat strain = 0;

    // keep track of the maximums for yield/fail/nonlinear materials
    // (and the ratio of the maximum from 0 to 1 [all positive end
    // strain to all negative end strain])
    Vfloat max_strain = 0, strain_offset = 0;

    // youngs modulus ratio between the positive end and the negative end
    // = E_Pos / E_neg
    Vfloat strain_ratio = 0;

    // pos1 is always = 0,0,0
    Vec3f pos2, angle1v, angle2v;

    // this bond in local coordinates.
    Quat3f angle1, angle2;

    // based on compiled precision setting
    // whether link is currently operating with a small angle assumption.
    bool is_small_angle = false;
    Vfloat current_rest_length = 0;

    // so we don't have to re-calculate everytime
    Vfloat current_transverse_area = 0, current_transverse_strain_sum = 0;

    // keep this around for convenience
    // the current overall true axial stress (MPa) between the two voxels.
    Vfloat axial_stress = 0;

    // Brand New Link, just after attachment
    int is_new_link = 0;
    bool is_detached = false;

    // for Secondary Experiment
    bool removed = false;

private:
    /*****************************************************************
     * Device side methods
     *****************************************************************/
    //!< updates pos2, angle1, angle2, and smallAngle.
    //!< returns the rotation quaternion (after
    //!< toAxisX) used to get to this orientation
    __device__ static Quat3f orientLink(VX3_Context &ctx, Vindex link);

    //!< Updates the rest length of this voxel. Call this
    //!< every timestep where the nominal size of either
    //!< voxel may have changed, due to actuation or thermal
    //!< expansion.
    __device__ static void updateRestLength(VX3_Context &ctx, Vindex link);

    //!< Updates information about this voxel pertaining
    //!< to volumetric deformations. Call this every
    //!< timestep if the poisson's ratio of the link
    //!< material is non-zero.
    __device__ static void updateTransverseInfo(VX3_Context &ctx, Vindex link);

    //!< Called every timestep to calculate the forces and
    //!< moments acting between the two constituent voxels in
    //!< their current relative positions and orientations.
    __device__ static void updateForces(VX3_Context &ctx, Vindex link);

    //!< updates strainNeg and strainPos
    //!< according to the provided axial strain.
    //!< returns current stress as well (MPa)
    __device__ static Vfloat updateStrain(VX3_Context &ctx, Vindex link,
                                          Vfloat axial_strain);

    __device__ static void setBoolState(VX3_Context &ctx, Vindex link, LinkFlags flag,
                                        bool active = true);
};

REFL_AUTO(type(VX3_Link), field(voxel_neg), field(voxel_pos), field(force_neg),
          field(force_pos), field(moment_neg), field(moment_pos), field(strain),
          field(max_strain), field(strain_offset), field(bool_states), field(axis),
          field(link_material), field(strain_ratio), field(pos2), field(angle1v),
          field(angle2v), field(angle1), field(angle2), field(is_small_angle),
          field(current_rest_length), field(current_transverse_area),
          field(current_transverse_strain_sum), field(axial_stress), field(is_new_link),
          field(is_detached), field(removed))

#endif // VX3_LINK_H
