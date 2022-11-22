#ifndef VX3_MATERIAL_H
#define VX3_MATERIAL_H

#include "utils/vx3_cuda.cuh"
#include "utils/vx3_def.h"
#include "utils/vx3_vec3d.h"
#include <string>
#include <vector>

#define MAX_MATERIAL_DATA_POINTS 16

struct VX3_InitContext;
struct VX3_Context;

class VX3_Material {
  public:
    VX3_Material() = default;

    //!< Initialize, check pre-set values, and updates all the derived
    //!< quantities cached as member variables.
    //!< This function is called by the kernel manager.
    virtual void init(const VX3_InitContext &ictx);

    //!< returns the stress of the material model accounting for
    //!< volumetric strain effects.
    //!< @param [in] strain The strain to query. The resulting stress
    //!< in this direction will be returned.
    //!< @param [in] transverse_strain_sum The sum of the two principle
    //!< normal strains in the plane perpendicular to strain.
    //!< @param [in] force_linear If true, the result will be calculated
    //!< according to the elastic modulus of the material regardless of
    //!< non-linearities in the model.
    template <bool isVoxelMaterial>
    __device__ static Vfloat stress(const VX3_Context &ctx, Vindex material,
                                    Vfloat strain, Vfloat transverse_strain_sum = 0.0f,
                                    bool force_linear = false);

    //!< Returns a simple reverse lookup of the first strain that
    //!< yields this stress from data point lookup.
    template <bool isVoxelMaterial>
    __device__ static Vfloat strain(const VX3_Context &ctx, Vindex material,
                                    Vfloat stress);

    //!< returns the modulus (slope of the stress/strain curve) of
    //!< the material model at the specified strain. @param [in]
    //!< strain The strain to query.
    template <bool isVoxelMaterial>
    __device__ static Vfloat modulus(const VX3_Context &ctx, Vindex material,
                                     Vfloat strain);

    //!< Returns true if the specified strain is past the yield point (if one is
    //!< specified). @param [in] strain The strain to query.
    template <bool isVoxelMaterial>
    __device__ static bool isYielded(const VX3_Context &ctx, Vindex material,
                                     Vfloat strain);

    //!< Returns true if the specified strain is past the failure point (if one is
    //!< specified). @param [in] strain The strain to query.
    template <bool isVoxelMaterial>
    __device__ static bool isFailed(const VX3_Context &ctx, Vindex material,
                                    Vfloat strain);

    //!< Returns true if poisson's ratio is zero - i.e. deformations in each dimension
    //!< are independent of those in other dimensions.
    template <bool isVoxelMaterial>
    __device__ static bool isXyzIndependent(const VX3_Context &ctx, Vindex material);

    /**
     * Pre-set attributes
     */

    int r = 0; //!< Red color value of this material from 0-255. Default is -1
    //!< (invalid/not set).
    int g = 0; //!< Green color value of this material from 0-255. Default is -1
    //!< (invalid/not set).
    int b = 0; //!< Blue color value of this material from 0-255. Default is -1
    //!< (invalid/not set).
    int a = 0; //!< Alpha value of this material from 0-255. Default is -1
    //!< (invalid/not set).

    int material_id = 0;

    // material model
    bool fixed = false;
    bool sticky = false;
    Vfloat cilia = 0;   // unused
    bool linear = true; //!< Set to true if this material is specified as linear.

    // All physical variables, name begins with their physical quantity
    // and end with meaning
    Vfloat E = 0;              //!< Young's modulus (stiffness) in Pa.
    Vfloat nu = 0;             //!< Poissons Ratio
    Vfloat rho = 0;            //!< Density in Kg/m^3
    Vfloat alpha_CTE = 0;      //!< Coefficient of thermal expansion (CTE)
    Vfloat fail_stress = -1;   //!< Failure stress in Pa. -1 indicates no failure
    Vfloat u_static = 0;       //!< Static coefficient of friction
    Vfloat u_kinetic = 0;      //!< Kinetic coefficient of friction
    Vfloat zeta_internal = 0;  //!< Internal damping ratio
    Vfloat zeta_collision = 0; //!< Collision damping ratio
    Vfloat zeta_global = 0;    //!< Global damping ratio

    bool is_target = false;
    bool is_measured = true;
    bool is_pace_maker = false;
    Vfloat pace_maker_period = 0;
    Vfloat signal_value_decay = 0.9; // ratio from [0,1]
    Vfloat signal_time_delay = 0.0;  // in sec
    Vfloat inactive_period = 0.05;   // in sec

    /**
     * Post-set attributes
     */
    Vfloat sigma_yield = 0;   //!< Yield stress in Pa.
    Vfloat sigma_fail = 0;    //!< Failure stress in Pa
    Vfloat epsilon_yield = 0; //!< Yield strain
    Vfloat epsilon_fail = 0;  //!< Failure strain
    //!< Effective elastic modulus for materials with non-zero Poisson's ratio.
    Vfloat E_hat = 0;

    /**
     * The arrays are assumed to be of equal length.
     * The first data point is assumed to be [0,0] and need not be provided.
     * At least 1 non-zero data point must be provided.
     * The inital segment from [0,0] to the first strain and stress value is
     * interpreted as young's modulus.
     * The slope of the stress/strain curve should never exceed this value in
     * subsequent segments.
     * The last data point is assumed to represent failure of the material.
     * The 0.2% offset method is used to calculate the yield point.
     *
     * Restrictions on pStrainValues:
        - The values must be positive and increasing in order.
        - Strains are defined in absolute numbers according to delta l / L.

     * Restrictions on pStressValues:
        - The values must be positive and increasing in order.

     * Special cases:
        - 1 data point (linear): Yield and failure are assumed to occur
          simultaneously at the single data point.
        - 2 data points (bilinear): Yield is taken as the first data point,
          failure at the second.
     */
    //!< Number of datapoints in strain/stress data
    int data_num = 0;
    Vfloat strain_data[MAX_MATERIAL_DATA_POINTS] = {0};
    Vfloat stress_data[MAX_MATERIAL_DATA_POINTS] = {0};

  private:
    //!< Initialize a linear material.
    //!< Note: when fail_stress is -1, indicates failure is not an option.
    // Used by init()
    void initModelLinear();

    //!< Initialize a Bilinear material.
    //!< Note: when fail_stress is -1, indicates failure is not an option.
    // Used by init()
    // unused
    // void initModelBilinear();
};

REFL_AUTO(type(VX3_Material), field(r), field(g), field(b), field(a), field(material_id),
          field(fixed), field(sticky), field(cilia), field(linear), field(E), field(nu),
          field(rho), field(alpha_CTE), field(u_static), field(u_kinetic),
          field(zeta_internal), field(zeta_global), field(zeta_collision),
          field(is_target), field(is_measured), field(is_pace_maker),
          field(pace_maker_period), field(signal_value_decay), field(signal_time_delay),
          field(inactive_period), field(sigma_yield), field(sigma_fail),
          field(epsilon_yield), field(epsilon_fail), field(E_hat), field(data_num),
          field(strain_data), field(stress_data))

#endif // VX3_MATERIAL_H
