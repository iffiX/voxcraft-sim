#ifndef VX3_VOXEL_MATERIAL_H
#define VX3_VOXEL_MATERIAL_H

#include "utils/vx3_cuda.cuh"
#include "utils/vx3_def.h"
#include "vx3/vx3_material.h"

class VX3_VoxelMaterial : public VX3_Material {
  public:
    VX3_VoxelMaterial() = default;

    //!< Initialize, check pre-set values, and updates all the derived
    //!< quantities cached as member variables.
    //!< This function is called by the kernel manager.
    void init(const VX3_InitContext &ictx) override;

    // damping convenience functions
    //!< Returns the internal material damping coefficient (translation).
    __device__ static Vfloat internalDampingTranslateC(const VX3_Context &ctx,
                                                       Vindex material);
    //!< Returns the internal material damping coefficient (rotation).
    __device__ static Vfloat internalDampingRotateC(const VX3_Context &ctx,
                                                    Vindex material);
    //!< Returns the global material damping coefficient (translation)
    __device__ static Vfloat globalDampingTranslateC(const VX3_Context &ctx,
                                                     Vindex material);
    //!< Returns the global material damping coefficient (rotation)
    __device__ static Vfloat globalDampingRotateC(const VX3_Context &ctx,
                                                  Vindex material);
    //!< Returns the global material damping coefficient (translation)
    __device__ static Vfloat collisionDampingTranslateC(const VX3_Context &ctx,
                                                        Vindex material);
    //!< Returns the global material damping coefficient (rotation)
    __device__ static Vfloat collisionDampingRotateC(const VX3_Context &ctx,
                                                     Vindex material);

    // stiffness
    //!< returns the stiffness with which this voxel will resist penetration.
    //!< This is calculated according to E*A/L with L = voxelSize/2.
    __device__ static Vfloat penetrationStiffness(const VX3_Context &ctx,
                                                  Vindex material);

    //!< Returns the current gravitational force on this voxel according to F=ma.
    __device__ static Vfloat gravityForce(const VX3_Context &ctx, Vindex material,
                                          Vfloat grav_acc = -9.80665);

    /**
     * Pre-set attributes
     */
    Vfloat nom_size = 0; //!< Nominal size (i.e. lattice dimension) (m)

    /**
     * Post-set attributes
     */
    Vfloat mass = 0;           //!< Cached mass of this voxel (kg)
    Vfloat mass_inverse = 0;   //!< Cached 1/Mass (1/kg)
    Vfloat sqrt_mass = 0;      //!< Cached sqrt(mass). (sqrt(Kg))
    Vfloat first_moment = 0;   //!< Cached 1st moment "inertia" (needed for certain
                               //!< calculations) (kg*m)
    Vfloat moment_inertia = 0; //!< Cached mass moment of inertia (i.e. rotational
                               //!< "mass") (kg*m^2)
    Vfloat moment_inertia_inverse = 0; //!< Cached 1/Inertia (1/(kg*m^2))
    Vfloat _2xSqMxExS = 0;     //!< Cached value needed for quick damping calculations
                               //!< (Kg*m/s)
    Vfloat _2xSqIxExSxSxS = 0; //!< Cached value needed for quick rotational damping
                               //!< calculations (Kg*m^2/s)
};

REFL_AUTO(type(VX3_VoxelMaterial),
          // Base fields
          field(r), field(g), field(b), field(a), field(material_id), field(fixed),
          field(sticky), field(cilia), field(linear), field(E), field(nu), field(rho),
          field(alpha_CTE), field(u_static), field(u_kinetic), field(zeta_internal),
          field(zeta_global), field(zeta_collision), field(is_target), field(is_measured),
          field(is_pace_maker), field(pace_maker_period), field(signal_value_decay),
          field(signal_time_delay), field(inactive_period), field(sigma_yield),
          field(sigma_fail), field(epsilon_yield), field(epsilon_fail), field(E_hat),
          field(data_num), field(strain_data), field(stress_data),
          // self fields
          field(nom_size), field(mass), field(mass_inverse), field(sqrt_mass),
          field(first_moment), field(moment_inertia), field(moment_inertia_inverse),
          field(_2xSqMxExS), field(_2xSqIxExSxSxS))

#endif // VX3_VOXEL_MATERIAL_H
