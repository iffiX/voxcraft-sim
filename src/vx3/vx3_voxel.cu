#include "vx3_voxel.h"

#include "vx3_link.h"
#include "vx3_voxel_material.h"
#include "vx3_voxelyze_kernel.cuh"

void VX3_Voxel::init(const VX3_InitContext &ictx) {
    // Currently no need to compute anything
}

__device__ void VX3_Voxel::timeStep(VX3_VoxelyzeKernel &k, Vindex voxel, Vfloat dt,
                                    Vfloat current_time) {
    auto &ctx = k.ctx;
    // http://klas-physics.googlecode.com/svn/trunk/src/general/Integrator.cpp (reference)
    V_S(previous_dt, dt);
    if (dt == VF(0.0))
        return;

    // Translation
    Vec3f cur_force = force(ctx, voxel, k.grav_acc);
    // Not implemented
    //    // Clear contact force
    //    V_S(contact_force, Vec3f());
    //    // Clear cilia force
    //    V_S(cilia_force, Vec3f());

    // Apply Force Field
    auto position = V_G(position);
    cur_force.x += k.force_field.x_forcefield(
        position.x, position.y, position.z, (Vfloat)k.collision_count, current_time,
        k.recent_angle, k.target_closeness, k.num_close_pairs, (int)ctx.voxels.size());
    cur_force.y += k.force_field.y_forcefield(
        position.x, position.y, position.z, (Vfloat)k.collision_count, current_time,
        k.recent_angle, k.target_closeness, k.num_close_pairs, (int)ctx.voxels.size());
    cur_force.z += k.force_field.z_forcefield(
        position.x, position.y, position.z, (Vfloat)k.collision_count, current_time,
        k.recent_angle, k.target_closeness, k.num_close_pairs, (int)ctx.voxels.size());

    Vec3f fric_force = cur_force;

    if (k.floor_enabled) {
        // floor force needs dt to calculate threshold to
        // "stop" a slow voxel into static friction.
        floorForce(ctx, voxel, dt, cur_force);
    }
    fric_force = cur_force - fric_force;

    // assert non QNAN
    assert(not isnan(cur_force.x) && not isnan(cur_force.y) && not isnan(cur_force.z));

    Vec3f _linear_momentum = V_G(linear_momentum);
    _linear_momentum += cur_force * dt;
    V_S(linear_momentum, _linear_momentum);

    // movement of the voxel this timestep
    Vec3f translate(_linear_momentum * (dt * VM_G(V_G(voxel_material), mass_inverse)));

    // we need to check for friction conditions here (after calculating the translation)
    // and stop things accordingly
    if (k.floor_enabled && floorPenetration(ctx, voxel) >= 0) {
        // we must catch a slowing voxel here since it all boils down to needing
        // access to the dt of this timestep.

        // F dot disp
        Vfloat work = fric_force.x * translate.x + fric_force.y * translate.y;

        // horizontal kinetic energy
        Vfloat hKe = VF(0.5) * VM_G(V_G(voxel_material), mass_inverse) *
                     (_linear_momentum.x * _linear_momentum.x +
                      _linear_momentum.y * _linear_momentum.y);
        if (hKe + work <= 0) {
            // this checks for a change of direction
            // according to the work-energy principle
            setBoolState(ctx, voxel, FLOOR_STATIC_FRICTION, true);
        }
        if (getBoolState(ctx, voxel, FLOOR_STATIC_FRICTION)) {
            // if we're in a state of static friction, zero out all horizontal motion
            _linear_momentum.x = _linear_momentum.y = 0;
            V_S(linear_momentum, _linear_momentum);
            translate.x = translate.y = 0;
        }
    } else
        setBoolState(ctx, voxel, FLOOR_STATIC_FRICTION, false);

    V_S(position, V_G(position) + translate);
    // Rotation
    Vec3f current_momentum = moment(ctx, voxel);
    V_S(angular_momentum, V_G(angular_momentum) + current_momentum * dt);

    // update the orientation
    V_S(orientation, Quat3f(V_G(angular_momentum) *
                            (dt * VM_G(V_G(voxel_material), moment_inertia_inverse))) *
                         V_G(orientation));

    //	we need to check for friction conditions here (after calculating the translation)
    // and stop things accordingly
    if (k.floor_enabled && floorPenetration(ctx, voxel) >= 0) {
        // we must catch a slowing voxel here since it all boils down to needing access to
        // the dt of this timestep.
        if (getBoolState(ctx, voxel, FLOOR_STATIC_FRICTION)) {
            V_S(angular_momentum, Vec3f());
        }
    }

    //    if (k.enable_signals) {
    //        propagateSignal(ctx, voxel, current_time);
    //        packMaker(ctx, voxel, current_time);
    //        localSignalDecay(ctx, voxel, current_time);
    //    }
    V_S(poissons_strain, strain(ctx, voxel, true));
    V_S(nnn_offset, cornerOffset(ctx, voxel, NNN));
    V_S(ppp_offset, cornerOffset(ctx, voxel, PPP));
}

__device__ void VX3_Voxel::updateTemperature(VX3_Context &ctx, Vindex voxel,
                                             Vfloat temperature) {
    V_S(temperature, temperature);
}

__device__ Vec3f VX3_Voxel::baseSize(const VX3_Context &ctx, Vindex voxel) {
    Vindex voxel_material = V_G(voxel_material);
    Vfloat nom_size = VM_G(voxel_material, nom_size);
    Vfloat alpha_CTE = VM_G(voxel_material, alpha_CTE);
    auto base_size = Vec3f(nom_size, nom_size, nom_size);
    return base_size * (1 + V_G(temperature) * alpha_CTE);
}

__device__ Vec3f VX3_Voxel::cornerPosition(const VX3_Context &ctx, Vindex voxel,
                                           VoxelCorner corner) {
    return V_G(position) + V_G(orientation).rotateVec3D(cornerOffset(ctx, voxel, corner));
}

__device__ Vec3f VX3_Voxel::cornerOffset(const VX3_Context &ctx, Vindex voxel,
                                         VoxelCorner corner) {
    Vec3f strains;
    auto links = V_G(links);
    for (int i = 0; i < 3; i++) {
        bool pos_link = corner & (1 << (2 - i)) ? true : false;
        Vindex link = links[2 * i + (pos_link ? 0 : 1)];
        if (link != NULL_INDEX && not VX3_Link::isFailed(ctx, link)) {
            strains[i] = (VF(1.0) + VX3_Link::axialStrain(ctx, link, pos_link)) *
                         (pos_link ? VF(1.0) : VF(-1.0));
        } else
            strains[i] = pos_link ? VF(1.0) : VF(-1.0);
    }

    return Vec3f((VF(0.5) * baseSize(ctx, voxel)).scale(strains));
}

__device__ Vfloat VX3_Voxel::transverseArea(const VX3_Context &ctx, Vindex voxel,
                                            LinkAxis axis) {
    Vindex voxel_material = V_G(voxel_material);
    Vfloat size = VM_G(voxel_material, nom_size);
    if (VM_G(voxel_material, nu) == 0)
        return size * size;

    Vec3f ps = V_G(poissons_strain);

    switch (axis) {
    case X_AXIS:
        return size * size * (VF(1) + ps.y) * (VF(1) + ps.z);
    case Y_AXIS:
        return size * size * (VF(1) + ps.x) * (VF(1) + ps.z);
    case Z_AXIS:
        return size * size * (VF(1) + ps.x) * (VF(1) + ps.y);
    default:
        return size * size;
    }
}

__device__ Vfloat VX3_Voxel::transverseStrainSum(const VX3_Context &ctx, Vindex voxel,
                                                 LinkAxis axis) {
    if (VM_G(V_G(voxel_material), nu) == 0)
        return 0;

    Vec3f ps = V_G(poissons_strain);

    switch (axis) {
    case X_AXIS:
        return ps.y + ps.z;
    case Y_AXIS:
        return ps.x + ps.z;
    case Z_AXIS:
        return ps.x + ps.y;
    default:
        return VF(0.0);
    }
}

__device__ Vfloat VX3_Voxel::floorPenetration(const VX3_Context &ctx, Vindex voxel) {
    Vindex voxel_material = V_G(voxel_material);
    Vfloat nom_size = VM_G(voxel_material, nom_size);
    Vfloat z = V_G(position).z;
    return baseSizeAverage(ctx, voxel) / 2 - nom_size / 2 - z;
}

__device__ Vec3f VX3_Voxel::force(const VX3_Context &ctx, Vindex voxel, Vfloat grav_acc) {
    Vindex voxel_material = V_G(voxel_material);

    // forces from internal bonds
    Vec3f total_force;
    auto links = V_G(links);
    for (int i = 0; i < 6; i++) {
        Vindex link = links[i];
        if (link != NULL_INDEX) {
            if (isLinkDirectionNegative((LinkDirection)i))
                total_force += L_G(link, force_neg);
            else
                total_force += L_G(link, force_pos);
        }
    }

    // from local to global coordinates
    total_force = V_G(orientation).rotateVec3D(total_force);

    // assert non QNAN
    assert(not isnan(total_force.x) && not isnan(total_force.y) &&
           not isnan(total_force.z));

    // other forces
    // global damping f-cv
    total_force -= velocity(ctx, voxel) *
                   VX3_VoxelMaterial::globalDampingTranslateC(ctx, voxel_material);

    // gravity, according to f=mg
    total_force.z += VX3_VoxelMaterial::gravityForce(ctx, voxel_material, grav_acc);

    // Not implemented
    //    total_force -= V_G(contact_force);
    //    total_force += V_G(cilia_force) * VM_G(voxel_material, cilia);
    return total_force;
}

__device__ Vec3f VX3_Voxel::moment(const VX3_Context &ctx, Vindex voxel) {
    Vindex voxel_material = V_G(voxel_material);

    // moments from internal bonds
    Vec3f total_moment(0, 0, 0);
    auto links = V_G(links);
    for (int i = 0; i < 6; i++) {
        Vindex link = links[i];
        if (link != NULL_INDEX) {
            if (isLinkDirectionNegative((LinkDirection)i))
                total_moment += L_G(link, moment_neg);
            else
                total_moment += L_G(link, moment_pos);
        }
    }
    total_moment = V_G(orientation).rotateVec3D(total_moment);

    // other moments
    // global damping
    total_moment -= angularVelocity(ctx, voxel) *
                    VX3_VoxelMaterial::globalDampingRotateC(ctx, voxel_material);

    return total_moment;
}

__device__ Vec3f VX3_Voxel::strain(const VX3_Context &ctx, Vindex voxel,
                                   bool poissons_strain) {
    // if no connections in the positive and negative directions of a particular axis,
    // strain is zero.
    // if one connection in positive or negative direction of a particular
    // axis, strain is that strain - ?? and force or constraint?
    // if connections in both the positive and negative directions of a particular axis,
    // strain is the average.

    // Per axis strain sum
    Vec3f axis_strain_sum;

    // number of bonds in this axis (0,1,2). axes according to LinkAxis enum
    int num_bond_axis[3] = {0};
    bool tension[3] = {false};

    auto links = V_G(links);
    for (int i = 0; i < 6; i++) { // cycle through link directions
        Vindex link = links[i];
        if (link != NULL_INDEX) {
            int axis = linkDirectionToAxis((LinkDirection)i);
            axis_strain_sum[axis] += VX3_Link::axialStrain(
                ctx, link, isLinkDirectionNegative((LinkDirection)i));
            num_bond_axis[axis]++;
        }
    }
    for (int i = 0; i < 3; i++) { // cycle through axes
        if (num_bond_axis[i] == 2)
            axis_strain_sum[i] *= VF(0.5); // average
        if (poissons_strain) {
            // if both sides pulling,
            // or just one side and a fixed or forced voxel... (not implemented)
            tension[i] = num_bond_axis[i] == 2;
        }
    }

    if (poissons_strain) {
        if (!(tension[0] && tension[1] && tension[2])) {
            // if at least one isn't in tension
            float add = 0;
            for (int i = 0; i < 3; i++)
                if (tension[i])
                    add += axis_strain_sum[i];
            float value = pow(VF(1.0) + add, -VM_G(V_G(voxel_material), nu)) - VF(1.0);
            for (int i = 0; i < 3; i++)
                if (!tension[i])
                    axis_strain_sum[i] = value;
        }
    }
    return axis_strain_sum;
}

__device__ Vec3f VX3_Voxel::velocity(const VX3_Context &ctx, Vindex voxel) {
    return V_G(linear_momentum) * VM_G(V_G(voxel_material), mass_inverse);
}

__device__ Vec3f VX3_Voxel::angularVelocity(const VX3_Context &ctx, Vindex voxel) {
    return V_G(angular_momentum) * VM_G(V_G(voxel_material), moment_inertia_inverse);
}

__device__ Vfloat VX3_Voxel::dampingMultiplier(const VX3_Context &ctx, Vindex voxel) {
    Vindex voxel_material = V_G(voxel_material);
    return 2 * VM_G(voxel_material, sqrt_mass) * VM_G(voxel_material, zeta_internal) /
           V_G(previous_dt);
}

__device__ bool VX3_Voxel::isSurface(const VX3_Context &ctx, Vindex voxel) {
    auto links = V_G(links);
    for (int i = 0; i < 6; i++)
        if (links[i] == NULL_INDEX)
            return true;
    return false;
}

__device__ bool VX3_Voxel::getBoolState(const VX3_Context &ctx, Vindex voxel,
                                        VoxFlags flag) {
    return V_G(bool_states) & flag ? true : false;
}

__device__ void VX3_Voxel::floorForce(VX3_Context &ctx, Vindex voxel, float dt,
                                      Vec3f &total_force) {
    // for now use the average.
    Vfloat current_pen = floorPenetration(ctx, voxel);
    Vindex voxel_material = V_G(voxel_material);
    if (current_pen >= 0) {
        Vec3f vel = velocity(ctx, voxel);
        Vec3f horizontal_vel(vel.x, vel.y, 0);

        Vfloat normal_force =
            VX3_VoxelMaterial::penetrationStiffness(ctx, voxel_material) * current_pen;

        // in the z direction: k*x-C*v - spring and damping
        total_force.z +=
            normal_force -
            VX3_VoxelMaterial::collisionDampingTranslateC(ctx, voxel_material) * vel.z;

        if (getBoolState(ctx, voxel, FLOOR_STATIC_FRICTION)) {
            // If this voxel is currently in static friction mode (no lateral motion)
            assert(horizontal_vel.length2() == 0);

            // use squares to avoid a square root
            Vfloat surface_force_sq =
                total_force.x * total_force.x + total_force.y * total_force.y;
            Vfloat friction_force_sq = (VM_G(voxel_material, u_static) * normal_force);
            friction_force_sq = friction_force_sq * friction_force_sq;

            // if we're breaking static friction, leave the forces as they
            // currently have been calculated to initiate motion this time
            // step
            if (surface_force_sq > friction_force_sq)
                setBoolState(ctx, voxel, FLOOR_STATIC_FRICTION, false);
        } else {
            // even if we just transitioned don't process here or else with a
            // complete lack of momentum it'll just go back to static friction

            // add a friction force opposing velocity
            // according to the normal force and the
            // kinetic coefficient of friction
            total_force -= VM_G(voxel_material, u_kinetic) * normal_force *
                           horizontal_vel.normalized();
        }
    } else
        setBoolState(ctx, voxel, FLOOR_STATIC_FRICTION, false);
}

__device__ void VX3_Voxel::setBoolState(VX3_Context &ctx, Vindex voxel, VoxFlags flag,
                                        bool active) {
    active ? V_S(bool_states, V_G(bool_states) | flag)
           : V_S(bool_states, V_G(bool_states) & ~flag);
}