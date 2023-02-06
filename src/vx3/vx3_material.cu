#include "vx3_material.h"

// Must include this at end because we need to define
// reflection macros in vx3_link, vx3_voxel, etc. first
#include "vx3/vx3_context.h"

#define MG(property) ((isVoxelMaterial) ? VM_G(property) : LM_G(property))

void VX3_Material::init(const VX3_InitContext &ictx) {
    // Bound pre-set values
    r = BOUND(r, 0, 255);
    g = BOUND(g, 0, 255);
    b = BOUND(b, 0, 255);
    a = BOUND(a, 0, 255);
    E = MAX(E, 0);
    // exactly 0.5 will still cause problems, but it can get very close.
    nu = BOUND(nu, 0, 0.5 - FLT_EPSILON * 2);
    rho = MAX(rho, 0);
    alpha_CTE = MAX(alpha_CTE, 0);
    u_static = MAX(u_static, 0);
    u_kinetic = MAX(u_kinetic, 0);
    zeta_internal = MAX(zeta_internal, 0);
    zeta_collision = MAX(zeta_collision, 0);
    zeta_global = MAX(zeta_global, 0);
    pace_maker_period = MAX(pace_maker_period, 0);
    signal_value_decay = BOUND(signal_value_decay, 0, 1);
    signal_time_delay = MAX(signal_time_delay, 0);
    inactive_period = MAX(inactive_period, 0);

    if (fail_stress < 0 and fail_stress != -1)
        throw std::invalid_argument("Failure stress is negative");

    //    if (linear)
    //        initModelLinear();
    //    else
    //        initModelBilinear();

    linear = true;
    initModelLinear();

    // update derived values
    E_hat = E / ((1 - 2 * nu) * (1 + nu));
}

template <bool isVoxelMaterial>
__device__ Vfloat VX3_Material::stress(const VX3_Context &ctx, Vindex material, Vfloat strain,
                                       Vfloat transverse_strain_sum, bool force_linear) {
    // reference:
    // http://www.colorado.edu/engineering/CAS/courses.d/Structures.d/IAST.Lect05.d/IAST.Lect05.pdf
    // page 10
    if (VX3_Material::isFailed<isVoxelMaterial>(ctx, material, strain))
        return 0.0f; // if a failure point is set and exceeded, we've broken!

    Vfloat E = MG(E);
    Vfloat nu = MG(nu);
    Vfloat E_hat = MG(E_hat);
    auto strain_data = MG(strain_data);
    if (strain <= strain_data[1] || MG(linear) || force_linear) {
        // for compression/first segment and linear materials (forced or otherwise),
        // simple calculation
        if (nu == 0.0f)
            return E * strain;
        else
            return E_hat * ((1 - nu) * strain + nu * transverse_strain_sum);
    }

    int data_num = MG(data_num);
    auto stress_data = MG(stress_data);

    // the non-linear feature with non-zero poissons ratio is currently experimental
    for (int i = 2; i < data_num; i++) {
        // go through each segment in the material model
        // (skipping the first segment because it has already been handled.
        if (strain <= strain_data[i] || i == data_num - 1) {
            // if in the segment ending with this point (or if this is the last point
            // extrapolate out)
            Vfloat perc =
                (strain - strain_data[i - 1]) / (strain_data[i] - strain_data[i - 1]);
            Vfloat basic_stress =
                stress_data[i - 1] + perc * (stress_data[i] - stress_data[i - 1]);
            if (nu == 0.0f)
                return basic_stress;
            else { // accounting for volumetric effects
                Vfloat modulus = (stress_data[i] - stress_data[i - 1]) /
                                (strain_data[i] - strain_data[i - 1]);
                Vfloat modulus_hat = modulus / ((1 - 2 * nu) * (1 + nu));
                // this is the strain at which a simple linear stress strain line
                // would hit this point at the definied modulus
                Vfloat effective_strain = basic_stress / modulus;
                Vfloat effective_transverse_strain_sum =
                    transverse_strain_sum * (effective_strain / strain);
                return modulus_hat * ((1 - nu) * effective_strain +
                                      nu * effective_transverse_strain_sum);
            }
        }
    }
    return 0.0f;
}
template __device__ Vfloat VX3_Material::stress<false>(const VX3_Context &, Vindex,
                                                       Vfloat, Vfloat, bool);
template __device__ Vfloat VX3_Material::stress<true>(const VX3_Context &, Vindex, Vfloat,
                                                      Vfloat, bool);

template <bool isVoxelMaterial>
__device__ Vfloat VX3_Material::strain(const VX3_Context &ctx, Vindex material, Vfloat stress) {
    auto stress_data = MG(stress_data);
    if (stress <= stress_data[1] || MG(linear))
        // for compression/first segment and linear materials (forced or otherwise),
        // simple calculation
        return stress / MG(E);

    int data_num = MG(data_num);
    auto strain_data = MG(strain_data);
    for (int i = 2; i < data_num; i++) {
        // go through each segment in the material model
        // (skipping the first segment because it has already been handled.
        if (stress <= stress_data[i] || i == data_num - 1) {
            // if in the segment ending with this point
            // (or if this is the last point extrapolate out)
            Vfloat perc =
                (stress - stress_data[i - 1]) / (stress_data[i] - stress_data[i - 1]);
            return strain_data[i - 1] + perc * (strain_data[i] - strain_data[i - 1]);
        }
    }
    return 0.0f;
}

template __device__ Vfloat VX3_Material::strain<false>(const VX3_Context &, Vindex,
                                                      Vfloat);
template __device__ Vfloat VX3_Material::strain<true>(const VX3_Context &, Vindex, Vfloat);

template <bool isVoxelMaterial>
__device__ Vfloat VX3_Material::modulus(const VX3_Context &ctx, Vindex material,
                                        Vfloat strain) {
    if (VX3_Material::isFailed<isVoxelMaterial>(ctx, material, strain))
        return 0.0f; // if a failure point is set and exceeded, we've broken!

    auto strain_data = MG(strain_data);
    if (strain <= strain_data[1] || MG(linear))
        return MG(
            E); // for compression/first segment and linear materials, simple calculation

    int data_num = MG(data_num);
    auto stress_data = MG(stress_data);
    for (int i = 2; i < data_num; i++) {
        // go through each segment in the material model
        // (skipping the first segment because it has already been handled.
        if (strain <= strain_data[i] || i == data_num - 1)
            // if in the segment ending with this point
            return (stress_data[i] - stress_data[i - 1]) /
                   (strain_data[i] - strain_data[i - 1]);
    }
    return 0.0f;
}
template __device__ Vfloat VX3_Material::modulus<false>(const VX3_Context &, Vindex,
                                                        Vfloat);
template __device__ Vfloat VX3_Material::modulus<true>(const VX3_Context &, Vindex,
                                                       Vfloat);

template <bool isVoxelMaterial>
__device__ bool VX3_Material::isYielded(const VX3_Context &ctx, Vindex material,
                                        Vfloat strain) {
    if constexpr (isVoxelMaterial) {
        Vfloat epsilon_yield = VM_G(epsilon_yield);
        return epsilon_yield != -1 && strain > epsilon_yield;
    } else {
        Vfloat epsilon_yield = LM_G(epsilon_yield);
        return epsilon_yield != -1 && strain > epsilon_yield;
    }
}
template __device__ bool VX3_Material::isYielded<false>(const VX3_Context &, Vindex,
                                                        Vfloat);
template __device__ bool VX3_Material::isYielded<true>(const VX3_Context &, Vindex,
                                                       Vfloat);

template <bool isVoxelMaterial>
__device__ bool VX3_Material::isFailed(const VX3_Context &ctx, Vindex material,
                                       Vfloat strain) {
    if constexpr (isVoxelMaterial) {
        Vfloat epsilon_fail = VM_G(epsilon_fail);
        return epsilon_fail != -1 && strain > epsilon_fail;
    } else {
        Vfloat epsilon_fail = LM_G(epsilon_fail);
        return epsilon_fail != -1 && strain > epsilon_fail;
    }
}
template __device__ bool VX3_Material::isFailed<false>(const VX3_Context &, Vindex,
                                                       Vfloat);
template __device__ bool VX3_Material::isFailed<true>(const VX3_Context &, Vindex,
                                                      Vfloat);

template <bool isVoxelMaterial>
__device__ bool VX3_Material::isXyzIndependent(const VX3_Context &ctx, Vindex material) {
    if constexpr (isVoxelMaterial) {
        return VM_G(nu) == 0;
    } else {
        return LM_G(nu) == 0;
    }
}
template __device__ bool VX3_Material::isXyzIndependent<false>(const VX3_Context &,
                                                               Vindex);
template __device__ bool VX3_Material::isXyzIndependent<true>(const VX3_Context &,
                                                              Vindex);

void VX3_Material::initModelLinear() {
    /**
     * Yield stress is interpreted as identical to failure stress.
     * If failure stress is not specified, an arbitrary data point
     * consistent with the specified Young's modulus is added to
     * the model.
     */

    // create a dummy failure stress if none was provided
    Vfloat tmp_fail_stress = fail_stress;
    if (tmp_fail_stress == -1)
        tmp_fail_stress = 1000000;
    Vfloat tmp_fail_strain = tmp_fail_stress / E;

    // add in the zero data point (required always)
    data_num = 2;
    strain_data[0] = 0;
    stress_data[0] = 0;
    strain_data[1] = tmp_fail_strain;
    stress_data[1] = tmp_fail_stress;

    sigma_yield = fail_stress; // yield and failure are one in the same here.
    sigma_fail = fail_stress;
    epsilon_yield = (fail_stress == -1) ? -1 : tmp_fail_strain;
    epsilon_fail = (fail_stress == -1) ? -1 : tmp_fail_strain;
}

// void VX3_Material::initModelBilinear() {
//    /**
//     * Specified Young's modulus, plastic modulus, yield stress,
//     * and failure stress must all be positive.
//     * Plastic modulus must be less than Young's modulus and
//     * failure stress must be greater than the yield stress.
//     */
//
//    // Unused, needs to define a plastic_modulus and a yield_stress member
//    // and add an entry in config etc.
//    if (plastic_modulus >= E) {
//        throw std::invalid_argument(
//                "Plastic modulus must be positive but less than Young's modulus");
//    }
//
//    float yield_strain = yield_stress / E;
//    // create a dummy failure stress if none was provided
//    float tmp_fail_stress = fail_stress;
//    if (tmp_fail_stress == -1)
//        tmp_fail_stress = 3 * yield_stress;
//
//    float tM = plastic_modulus;
//    float tB = yield_stress - tM * yield_strain;          // y-mx=b
//    float tmp_fail_strain = (tmp_failure_stress - tB) / tM; // (y-b)/m = x
//
//    data_num = 3;
//    strain_data[0] = 0; // add in the zero data point (required always)
//    strain_data[1] = yield_strain;
//    strain_data[2] = tmp_fail_strain;
//
//    stress_data[0] = 0; // add in the zero data point (required always)
//    stress_data[1] = yield_stress;
//    stress_data[2] = tmp_fail_stress;
//
//    sigma_yield = yield_stress;
//    sigma_fail = failure_stress;
//    epsilon_yield = yield_strain;
//    epsilon_fail = fail_stress == -1.0f ? -1.0f : tmp_fail_strain;
//}