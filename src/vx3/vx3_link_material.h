#ifndef VX3_MATERIAL_LINK_H
#define VX3_MATERIAL_LINK_H

#include "utils/vx3_cuda.cuh"
#include "utils/vx3_def.h"
#include "vx3/vx3_voxel_material.h"

class VX3_LinkMaterial : public VX3_VoxelMaterial {
  public:
    VX3_LinkMaterial() = default;

    //!< Initialize, check pre-set values, and updates all the derived
    //!< quantities cached as member variables.
    //!< This function is called by the kernel manager.
    void init(const VX3_InitContext &ictx) override;

    /**
     * Pre-set attributes
     */
    Vindex vox1_mat = NULL_INDEX; //!< Constituent material 1 from one voxel
    Vindex vox2_mat = NULL_INDEX; //!< Constituent material 2 from the other voxel

    /**
     * Post-set attributes
     */
    Vfloat a1 = 0;       //!< Cached a1 beam constant.
    Vfloat a2 = 0;       //!< Cached a2 beam constant.
    Vfloat b1 = 0;       //!< Cached b1 beam constant.
    Vfloat b2 = 0;       //!< Cached b2 beam constant.
    Vfloat b3 = 0;       //!< Cached b3 beam constant.
    Vfloat sqA1 = 0;     //!< Cached sqrt(a1) constant for damping calculations.
    Vfloat sqA2xIp = 0;  //!< Cached sqrt(a2*L*L/6) constant for damping calculations.
    Vfloat sqB1 = 0;     //!< Cached sqrt(b1) constant for damping calculations.
    Vfloat sqB2xFMp = 0; //!< Cached sqrt(b2*L/2) constant for damping calculations.
    Vfloat sqB3xIp = 0;  //!< Cached sqrt(b3*L*L/6) constant for damping calculations.
};

REFL_AUTO(type(VX3_LinkMaterial),
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
          field(vox1_mat), field(vox2_mat), field(a1), field(a2), field(b1), field(b2),
          field(b3), field(sqA1), field(sqA2xIp), field(sqB1), field(sqB2xFMp),
          field(sqB3xIp))

#endif // VX3_MATERIAL_LINK_H
