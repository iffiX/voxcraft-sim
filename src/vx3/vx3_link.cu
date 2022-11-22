#include "vx3/vx3_link_material.h"
#include "vx3/vx3_voxel.h"
#include "vx3/vx3_voxel_material.h"
#include "vx3/vx3_voxelyze_kernel.cuh"
#include "vx3_link.h"

// Must include this at end because we need to define
// reflection macros in vx3_link, vx3_voxel, etc. first
#include "vx3/vx3_context.h"

/**
 * Parameter naming:
 * _some_var: is temporary variable which stores new states that will be used
 *  to update the member variable;
 * some_var: is local copy of the existing states that will not be used
 *  to update the member variable;
 */

void VX3_Link::init(const VX3_InitContext &ictx) {
    bool_states = 0;
    force_neg = Vec3f();
    force_pos = Vec3f();
    moment_neg = Vec3f();
    moment_pos = Vec3f();
    strain = 0;
    max_strain = 0;
    strain_offset = 0;

    Vfloat E_pos = ictx.voxel_materials[ictx.voxels[voxel_pos].voxel_material].E;
    Vfloat E_neg = ictx.voxel_materials[ictx.voxels[voxel_neg].voxel_material].E;
    strain_ratio = E_pos / E_neg;

    pos2 = Vec3f();
    angle1v = Vec3f();
    angle2v = Vec3f();
    angle1 = Quat3f();
    angle2 = Quat3f();
    is_small_angle = false;
    current_rest_length = 0;
    current_transverse_area = 0;
    current_transverse_strain_sum = 0;
    axial_stress = 0;
    is_new_link = 0;
    is_detached = false;
    removed = false;
}

__device__ Vfloat VX3_Link::axialStrain(const VX3_Context &ctx, Vindex link,
                                        bool positive_end) {
    Vfloat s = L_G(strain);
    Vfloat sr = L_G(strain_ratio);
    return positive_end ? 2.0f * s * sr / (1.0f + sr) : 2.0f * s / (1.0f + sr);
}

__device__ bool VX3_Link::isYielded(const VX3_Context &ctx, Vindex link) {
    return VX3_Material::isYielded<false>(ctx, L_G(max_strain), L_G(link_material));
}

__device__ bool VX3_Link::isFailed(const VX3_Context &ctx, Vindex link) {
    return VX3_Material::isFailed<false>(ctx, L_G(link_material), L_G(max_strain));
}

__device__ void VX3_Link::updateRestLength(VX3_Context &ctx, Vindex link) {
    // update rest length according to temperature of both end
    auto axis = L_G(axis);
    auto neg_base_size = VX3_Voxel::baseSize(ctx, L_G(voxel_neg), L_G(axis));
    auto pos_base_size = VX3_Voxel::baseSize(ctx, L_G(voxel_neg), L_G(axis));
    L_S(current_rest_length, 0.5f * (neg_base_size + pos_base_size));
}

__device__ void VX3_Link::updateTransverseInfo(VX3_Context &ctx, Vindex link) {
    L_S(current_transverse_area,
        0.5f * (VX3_Voxel::transverseArea(ctx, L_G(voxel_neg), L_G(axis)) +
                VX3_Voxel::transverseArea(ctx, L_G(voxel_pos), L_G(axis))));
    L_S(current_transverse_strain_sum,
        0.5f * (VX3_Voxel::transverseStrainSum(ctx, L_G(voxel_neg), L_G(axis)) +
                VX3_Voxel::transverseStrainSum(ctx, L_G(voxel_pos), L_G(axis))));
}

// updates pos2, angle1, angle2, and smallAngle
__device__ Quat3f VX3_Link::orientLink(VX3_Context &ctx, Vindex link) {
    Vec3f _pos2 = V_G(L_G(voxel_pos), position) - V_G(L_G(voxel_neg), position);
    // digit truncation happens here...
    _pos2 = toAxisX(L_G(axis), _pos2);

    Quat3f _angle1 = V_G(L_G(voxel_neg), orientation);
    _angle1 = toAxisX(L_G(axis), _angle1);

    Quat3f _angle2 = V_G(L_G(voxel_pos), orientation);
    _angle2 = toAxisX(L_G(axis), _angle2);

    // keep track of the total rotation of this bond (after toAxisX())
    Quat3f total_rot = _angle1.conjugate();
    _pos2 = total_rot.rotateVec3D(_pos2);
    _angle1 = Quat3f(); // zero for now...
    _angle2 = total_rot * _angle2;

    // small angle approximation?
    Vfloat curr_rest_length = L_G(current_rest_length);
    Vfloat small_turn = (abs(_pos2.z) + abs(_pos2.y)) / _pos2.x;
    Vfloat extend_perc = abs(1 - _pos2.x / curr_rest_length);
    bool _is_small_angle = L_G(is_small_angle);
    if (!_is_small_angle && small_turn < SA_BOND_BEND_RAD &&
        extend_perc < SA_BOND_EXT_PERC) {
        _is_small_angle = true;
        setBoolState(ctx, link, LOCAL_VELOCITY_VALID, false);
    } else if (_is_small_angle && small_turn > HYSTERESIS_FACTOR * SA_BOND_BEND_RAD ||
               extend_perc > HYSTERESIS_FACTOR * SA_BOND_EXT_PERC) {
        _is_small_angle = false;
        setBoolState(ctx, link, LOCAL_VELOCITY_VALID, false);
    }

    if (_is_small_angle) {
        // Align so Angle1 is all zeros

        // only valid for small angles
        _pos2.x -= curr_rest_length;
    } else {
        // Large angle. Align so that Pos2.y, Pos2.z are zero.

        // get the angle to align Pos2 with the X axis
        _angle1.fromAngleToPosX(_pos2);
        total_rot = _angle1 * total_rot; // update our total rotation to reflect this
        _angle2 = _angle1 * _angle2;     // rotate angle2
        _pos2 = Vec3f(_pos2.length() - curr_rest_length, 0, 0);
    }
    // State updates
    L_S(is_small_angle, _is_small_angle);
    L_S(angle1, _angle1);
    L_S(angle2, _angle2);
    L_S(pos2, _pos2);
    L_S(angle1v, _angle1.toRotationVector());
    L_S(angle2v, _angle1.toRotationVector());
    Vec3f a1v = L_G(angle1v);
    Vec3f a2v = L_G(angle2v);

    // assert non QNAN
    // assert(not isnan(a1v.x) && not isnan(a1v.y) && not isnan(a1v.z));
    // assert(not isnan(a2v.x) && not isnan(a2v.y) && not isnan(a2v.z));

    return total_rot;
}

__device__ void VX3_Link::updateForces(VX3_Context &ctx, Vindex link) {
    // remember the positions/angles from last timestamp to
    // calculate velocity
    Vec3f old_pos2 = L_G(pos2);
    Vec3f old_angle1v = L_G(angle1v);
    Vec3f old_angle2v = L_G(angle2v);

    // sets pos2, angle1, angle2
    orientLink(ctx, link);

    // deltas for local damping. velocity at center
    // is half the total velocity
    Vec3f new_pos2 = L_G(pos2);
    Vec3f new_angle1v = L_G(angle1v);
    Vec3f new_angle2v = L_G(angle2v);
    Vec3f d_pos2 = 0.5 * (new_pos2 - old_pos2);
    Vec3f d_angle1 = 0.5 * (new_angle1v - old_angle1v);
    Vec3f d_angle2 = 0.5 * (new_angle2v - old_angle2v);

    // if volume effects..
    if (!VX3_Material::isXyzIndependent<false>(ctx, L_G(link_material)) ||
        L_G(current_transverse_strain_sum) != 0) {
        // current_transverse_strain_sum != 0 catches when we disable
        // poissons mid-simulation
        // updateTransverseInfo();
    }
    Vfloat _axial_stress = updateStrain(ctx, link, new_pos2.x / L_G(current_rest_length));
    L_S(axial_stress, _axial_stress);
    if (isFailed(ctx, link)) {
        L_S(force_neg, Vec3f());
        L_S(force_pos, Vec3f());
        L_S(force_neg, Vec3f());
        L_S(force_pos, Vec3f());
        return;
    }

    // local copies
    Vindex link_mat = L_G(link_material);
    Vfloat b1 = LM_G(link_mat, b1), b2 = LM_G(link_mat, b2), b3 = LM_G(link_mat, b3),
           a2 = LM_G(link_mat, a2);

    // Beam equations. All relevant terms are here, even though some are zero
    // for small angle and others are zero for large angle (profiled as
    // negligible performance penalty)

    // Use Curstress instead of -a1*Pos2.x to account for non-linear deformation
    Vec3f _force_neg = Vec3f(_axial_stress * L_G(current_transverse_area),
                             b1 * new_pos2.y - b2 * (new_angle1v.z + new_angle2v.z),
                             b1 * new_pos2.z + b2 * (new_angle1v.y + new_angle2v.y));

    Vec3f _moment_neg = Vec3f(a2 * (new_angle2v.x - new_angle1v.x),
                              -b2 * new_pos2.z - b3 * (2 * new_angle1v.y + new_angle2v.y),
                              b2 * new_pos2.y - b3 * (2 * new_angle1v.z + new_angle2v.z));
    Vec3f _moment_pos = Vec3f(a2 * (new_angle1v.x - new_angle2v.x),
                              -b2 * new_pos2.z - b3 * (new_angle1v.y + 2 * new_angle2v.y),
                              b2 * new_pos2.y - b3 * (new_angle1v.z + 2 * new_angle2v.z));
    L_S(force_neg, _force_neg);
    L_S(force_pos, -_force_neg);
    L_S(moment_neg, _moment_neg);
    L_S(moment_pos, _moment_pos);

    // local damping:
    if (getBoolState(ctx, link, LOCAL_VELOCITY_VALID)) {
        // if we don't have the basis for a good damping calculation,
        // don't do any damping.

        Vfloat sqA1 = LM_G(link_mat, sqA1), sqA2xIp = LM_G(link_mat, sqA2xIp),
               sqB1 = LM_G(link_mat, sqB1), sqB2xFMp = LM_G(link_mat, sqB2xFMp),
               sqB3xIp = LM_G(link_mat, sqB3xIp);

        Vec3f pos_calc(sqA1 * d_pos2.x,
                       sqB1 * d_pos2.y - sqB2xFMp * (d_angle1.z + d_angle2.z),
                       sqB1 * d_pos2.z + sqB2xFMp * (d_angle1.y + d_angle2.y));

        Vfloat voxel_neg_damping = VX3_Voxel::dampingMultiplier(ctx, L_G(voxel_neg));
        Vfloat voxel_pos_damping = VX3_Voxel::dampingMultiplier(ctx, L_G(voxel_pos));

        L_S(force_neg, L_G(force_neg) + voxel_neg_damping * pos_calc);
        L_S(force_pos, L_G(force_pos) + voxel_pos_damping * pos_calc);
        L_S(moment_neg,
            L_G(moment_neg) -
                0.5 * voxel_neg_damping *
                    Vec3f(-sqA2xIp * (d_angle2.x - d_angle1.x),
                          sqB2xFMp * d_pos2.z + sqB3xIp * (2 * d_angle1.y + d_angle2.y),
                          -sqB2xFMp * d_pos2.y +
                              sqB3xIp * (2 * d_angle1.z + d_angle2.z)));
        L_S(moment_pos,
            L_G(moment_pos) -
                0.5 * voxel_pos_damping *
                    Vec3f(sqA2xIp * (d_angle2.x - d_angle1.x),
                          sqB2xFMp * d_pos2.z + sqB3xIp * (d_angle1.y + 2 * d_angle2.y),
                          -sqB2xFMp * d_pos2.y +
                              sqB3xIp * (d_angle1.z + 2 * d_angle2.z)));

    } else {
        // we're good for next go-around unless something changes
        setBoolState(ctx, link, LOCAL_VELOCITY_VALID, true);
    }
    //	transform forces and moments to local voxel coordinates
    LinkAxis ax = L_G(axis);
    if (!L_G(is_small_angle)) {
        Quat3f ang1 = L_G(angle1);
        L_S(force_neg, toAxisOriginal(ax, ang1.rotateVec3DInv(L_G(force_neg))));
        L_S(moment_neg, toAxisOriginal(ax, ang1.rotateVec3DInv(L_G(moment_neg))));
    }

    Quat3f ang2 = L_G(angle2);
    L_S(force_neg, toAxisOriginal(ax, ang2.rotateVec3DInv(L_G(force_neg))));
    L_S(force_pos, toAxisOriginal(ax, ang2.rotateVec3DInv(L_G(force_pos))));
    L_S(moment_neg, toAxisOriginal(ax, ang2.rotateVec3DInv(L_G(moment_neg))));
    L_S(moment_pos, toAxisOriginal(ax, ang2.rotateVec3DInv(L_G(moment_pos))));

    if (L_G(is_new_link)) {
        // for debug
        L_S(force_neg, L_G(force_neg) * 0.01);
        L_S(force_pos, L_G(force_pos) * 0.01);
        L_S(moment_neg, L_G(moment_neg) * 0.01);
        L_S(moment_pos, L_G(moment_pos) * 0.01);
        L_S(is_new_link, L_G(is_new_link) - 1);
    }
}

__device__ float VX3_Link::updateStrain(VX3_Context &ctx, Vindex link,
                                        float axial_strain) {
    // int di = 0;
    // redundant?
    L_S(strain, axial_strain);

    Vindex link_material = L_G(link_material);
    if (LM_G(link_material, linear)) {
        // remember this maximum for easy reference
        if (axial_strain > L_G(max_strain))
            L_S(max_strain, axial_strain);
        return VX3_Material::stress<false>(ctx, link_material, axial_strain,
                                           L_G(current_transverse_strain_sum));
    } else {
        float return_stress;
        if (axial_strain > L_G(max_strain)) {
            // if new territory on the stress/strain curve

            // remember this maximum for easy reference
            L_S(max_strain, axial_strain);
            Vfloat max_strain = axial_strain;
            return_stress = VX3_Material::stress<false>(
                ctx, link_material, axial_strain, L_G(current_transverse_strain_sum));

            if (LM_G(link_material, nu) != 0.0f) {
                // precalculate strain offset for when we back off
                L_S(strain_offset,
                    max_strain -
                        VX3_Material::stress<false>(ctx, link_material, axial_strain) /
                            (LM_G(link_material, E_hat) * (1 - LM_G(link_material, nu))));
            } else {
                // Precalculate strain offset for when we back off
                L_S(strain_offset, max_strain - return_stress / LM_G(link_material, E));
            }

        } else {
            // backed off a non-linear material, therefore in linear
            // region.

            // treat the material as linear with
            // a strain offset according to the
            // maximum plastic deformation
            Vfloat relative_strain = axial_strain - L_G(strain_offset);

            if (LM_G(link_material, nu) != 0.0f)
                return_stress =
                    VX3_Material::stress<false>(ctx, link_material, relative_strain,
                                                L_G(current_transverse_strain_sum), true);
            else
                return_stress = LM_G(link_material, E) * relative_strain;
        }

        return return_stress;
    }
}

__device__ void VX3_Link::setBoolState(VX3_Context &ctx, Vindex link, LinkFlags flag,
                                       bool active) {
    active ? L_S(bool_states, L_G(bool_states) | (int)flag)
           : L_S(bool_states, L_G(bool_states) & ~(int)flag);
}

__device__ bool VX3_Link::getBoolState(const VX3_Context &ctx, Vindex link, LinkFlags flag) {
    return L_G(bool_states) & flag ? true : false;
}

__device__ Vfloat VX3_Link::strainEnergy(const VX3_Context &ctx, Vindex link) {
    Vec3f force_neg = L_G(force_neg);
    Vec3f moment_neg = L_G(moment_neg);
    Vec3f moment_pos = L_G(moment_pos);
    Vindex link_material = L_G(link_material);
    Vfloat a1 = LM_G(link_material, a1);
    Vfloat a2 = LM_G(link_material, a2);
    Vfloat b3 = LM_G(link_material, b3);
    return force_neg.x * force_neg.x / (2.0f * a1) +   // Tensile strain
           moment_neg.x * moment_neg.x / (2.0f * a2) + // Torsion strain
           (moment_neg.z * moment_neg.z - moment_neg.z * moment_pos.z +
            moment_pos.z * moment_pos.z) /
               (3.0f * b3) + // Bending Z
           (moment_neg.y * moment_neg.y - moment_neg.y * moment_pos.y +
            moment_pos.y * moment_pos.y) /
               (3.0f * b3); // Bending Y
}

__device__ Vfloat VX3_Link::axialStiffness(VX3_Context &ctx, Vindex link) {
    Vindex link_material = L_G(link_material);
    if (VX3_Material::isXyzIndependent<false>(ctx, link_material))
        return LM_G(link_material, a1);
    else {
        updateRestLength(ctx, link);
        updateTransverseInfo(ctx, link);
        return LM_G(link_material, E_hat) * L_G(current_transverse_area) /
               ((L_G(strain) + 1.0f) * L_G(current_rest_length)); // _a1;
    }
}