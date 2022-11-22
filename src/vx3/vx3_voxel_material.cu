#include "vx3_voxel_material.h"

// Must include this at end because we need to define
// reflection macros in vx3_link, vx3_voxel, etc. first
#include "vx3/vx3_context.h"

void VX3_VoxelMaterial::init(const VX3_InitContext &ictx) {
    nom_size = MAX(nom_size, FLT_MIN);
    VX3_Material::init(ictx);

    Vfloat volume = nom_size * nom_size * nom_size;
    mass = volume * rho;
    moment_inertia = mass * nom_size * nom_size / (Vfloat)6.0; // simple 1D approx
    first_moment = mass * nom_size / (Vfloat)2.0;

    if (volume == 0 || mass == 0 || moment_inertia == 0) {
        // zero everything out
        mass_inverse = sqrt_mass = moment_inertia_inverse = _2xSqMxExS = _2xSqIxExSxSxS =
            0.0;
    }

    mass_inverse = (Vfloat)1.0 / mass;
    sqrt_mass = sqrt(mass);
    moment_inertia_inverse = (Vfloat)1.0 / moment_inertia;
    _2xSqMxExS = (Vfloat)2.0 * sqrt(mass * E * nom_size);
    _2xSqIxExSxSxS =
        (Vfloat)2.0 * sqrt(moment_inertia * E * nom_size * nom_size * nom_size);
}

__device__ Vfloat VX3_VoxelMaterial::internalDampingTranslateC(const VX3_Context &ctx, Vindex material) {
    return VM_G(zeta_internal) * VM_G(_2xSqMxExS);
}
__device__ Vfloat VX3_VoxelMaterial::internalDampingRotateC(const VX3_Context &ctx, Vindex material) {
    return VM_G(zeta_internal) * VM_G(_2xSqIxExSxSxS);
}
__device__ Vfloat VX3_VoxelMaterial::globalDampingTranslateC(const VX3_Context &ctx, Vindex material) {
    return VM_G(zeta_global) * VM_G(_2xSqMxExS);
}
__device__ Vfloat VX3_VoxelMaterial::globalDampingRotateC(const VX3_Context &ctx, Vindex material) {
    return VM_G(zeta_global) * VM_G(_2xSqIxExSxSxS);
}
__device__ Vfloat VX3_VoxelMaterial::collisionDampingTranslateC(const VX3_Context &ctx, Vindex material) {
    return VM_G(zeta_collision) * VM_G(_2xSqMxExS);
}
__device__ Vfloat VX3_VoxelMaterial::collisionDampingRotateC(const VX3_Context &ctx, Vindex material) {
    return VM_G(zeta_collision) * VM_G(_2xSqIxExSxSxS);
}

__device__ Vfloat VX3_VoxelMaterial::penetrationStiffness(const VX3_Context &ctx,
                                                          Vindex material) {
    return (Vfloat)2.0 * VM_G(E) * VM_G(nom_size);
}

__device__ Vfloat VX3_VoxelMaterial::gravityForce(const VX3_Context &ctx, Vindex material, Vfloat grav_acc) {
    return VM_G(mass) * grav_acc;
}