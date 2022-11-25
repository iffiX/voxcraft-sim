//
// Created by iffi on 11/6/22.
//

#include "vx3_voxelyze_kernel_manager.cuh"
#include <boost/format.hpp>

using namespace std;
using namespace boost;

VX3_VoxelyzeKernel
VX3_VoxelyzeKernelManager::createKernelFromConfig(const VX3_Config &config,
                                                  const cudaStream_t &stream) {
    // Add the default "empty" material, then add user specified materials
    ictx.voxel_materials.emplace_back(VX3_VoxelMaterial());
    for (auto &material_config : config.palette.materials) {
        addVoxelMaterial(material_config, (int)ictx.voxel_materials.size(),
                         config.lattice.lattice_dim, config.bond_damping_z,
                         config.col_damping_z, config.slow_damping_z);
    }
    // voxels and links (links are added when creating voxels)
    addVoxels(config.structure, config.lattice.lattice_dim);

    // Initialize
    for (auto &voxel_material : ictx.voxel_materials)
        voxel_material.init(ictx);
    for (auto &link_material : ictx.link_materials)
        link_material.init(ictx);
    for (auto &voxel : ictx.voxels)
        voxel.init(ictx);
    for (auto &link : ictx.links)
        link.init(ictx);

    // Copy data to the voxelyze kernel
    VX3_VoxelyzeKernel kernel(stream);
    kernel.ctx.link_materials.fill(ictx.link_materials, stream);
    kernel.ctx.voxel_materials.fill(ictx.voxel_materials, stream);
    kernel.ctx.links.fill(ictx.links, stream);
    kernel.ctx.voxels.fill(ictx.voxels, stream);

    kernel.vox_size = config.lattice.lattice_dim;
    kernel.dt_frac = config.dt_frac;

    // In VXA.Simulator.StopCondition
    setMathExpression(kernel.stop_condition_formula, config.stop_condition_formula);

    // In VXA.Simulator.RecordHistory
    kernel.record_step_size = config.record_step_size;
    kernel.record_link = config.record_link;
    kernel.record_voxel = config.record_voxel;

    // In VXA.Simulator.AttachDetach
    kernel.enable_attach = config.enable_attach;
    kernel.enable_detach = config.enable_detach;
    kernel.enable_collision = config.enable_collision;
    kernel.bounding_radius = config.bounding_radius;
    kernel.watch_distance = config.watch_distance;
    kernel.safety_guard = config.safety_guard;
    for (size_t i = 0; i < 5; i++)
        setMathExpression(kernel.attach_conditions[i], config.attach_conditions[i]);

    // In VXA.Simulator.ForceField
    setMathExpression(kernel.force_field.x_force_field, config.x_force_field);
    setMathExpression(kernel.force_field.y_force_field, config.y_force_field);
    setMathExpression(kernel.force_field.z_force_field, config.z_force_field);

    // In VXA.Simulator
    setMathExpression(kernel.fitness_function, config.fitness_function);
    kernel.max_dist_in_voxel_lengths_to_count_as_pair =
        config.max_dist_in_voxel_lengths_to_count_as_pair;
    kernel.save_position_of_all_voxels = config.save_position_of_all_voxels;
    kernel.enable_cilia = config.enable_cilia;
    kernel.enable_signals = config.enable_signals;

    // In VXA.Environment.Gravity
    kernel.grav_enabled = config.grav_enabled;
    kernel.floor_enabled = config.floor_enabled;
    kernel.grav_acc = config.grav_acc;

    // In VXA.Environment.Thermal
    kernel.enable_vary_temp = config.enable_vary_temp;
    kernel.temp_amplitude = config.temp_amplitude;
    kernel.temp_period = config.temp_period;

    // Collect target voxel indices
    Vindex *tmp_target_indices;
    Vsize target_num = 0;
    VcudaMallocHost(&tmp_target_indices, sizeof(Vindex) * ictx.voxels.size());
    for (Vindex vox = 0; vox < ictx.voxels.size(); vox++) {
        if (ictx.voxel_materials[ictx.voxels[vox].voxel_material].is_target) {
            tmp_target_indices[target_num++] = vox;
        }
    }
    VcudaMallocAsync(&kernel.target_indices, sizeof(Vindex) * target_num, stream);
    VcudaMemcpyAsync(kernel.target_indices, tmp_target_indices,
                     sizeof(Vindex) * target_num, cudaMemcpyHostToDevice, stream);
    VcudaFreeHost(tmp_target_indices);
    kernel.target_num = target_num;

    // Setup reduce memory
    Vsize max_elem_num = MAX(ictx.voxels.size(), ictx.links.size());
    VcudaMallocHost(&kernel.h_reduce_output, sizeof(Vfloat) * 4);
    VcudaMallocAsync(&kernel.d_reduce1, sizeof(Vfloat) * max_elem_num * 4, stream);
    VcudaMallocAsync(&kernel.d_reduce2, sizeof(Vfloat) * max_elem_num * 4, stream);
    return kernel;
}

void VX3_VoxelyzeKernelManager::freeKernel(VX3_VoxelyzeKernel &kernel) {
    kernel.ctx.link_materials.free(kernel.stream);
    kernel.ctx.voxel_materials.free(kernel.stream);
    kernel.ctx.links.free(kernel.stream);
    kernel.ctx.voxels.free(kernel.stream);
    VcudaFreeHost(kernel.h_reduce_output);
    VcudaFreeAsync(kernel.d_reduce1, kernel.stream);
    VcudaFreeAsync(kernel.d_reduce2, kernel.stream);
    VcudaFreeAsync(kernel.d_steps, kernel.stream);
    VcudaFreeAsync(kernel.d_time_points, kernel.stream);
    VcudaFreeAsync(kernel.d_link_record, kernel.stream);
    VcudaFreeAsync(kernel.d_voxel_record, kernel.stream);
}

void VX3_VoxelyzeKernelManager::addVoxelMaterial(
    const VX3_PaletteMaterialConfig &material_config, int material_id, Vfloat lattice_dim,
    Vfloat internal_damping, Vfloat collision_damping, Vfloat global_damping) {
    VX3_VoxelMaterial mat;

    // Base material attributes
    mat.r = int(round(material_config.r * 255));
    mat.g = int(round(material_config.g * 255));
    mat.b = int(round(material_config.b * 255));
    mat.a = int(round(material_config.a * 255));

    mat.material_id = material_id;
    mat.fixed = material_config.fixed;
    mat.sticky = material_config.sticky;
    mat.cilia = material_config.cilia;
    mat.linear = material_config.mat_model == VX3_PaletteMaterialConfig::MAT_LINEAR;

    mat.E = material_config.elastic_mod;
    mat.nu = material_config.poissons_ratio;
    mat.rho = material_config.density;
    mat.alpha_CTE = material_config.CTE;
    mat.fail_stress = material_config.fail_stress;
    mat.u_static = material_config.u_static;
    mat.u_kinetic = material_config.u_dynamic;
    mat.zeta_internal = internal_damping;
    mat.zeta_collision = collision_damping;
    mat.zeta_global = global_damping;

    mat.is_target = material_config.is_target;
    mat.is_measured = material_config.is_measured;
    mat.is_pace_maker = material_config.is_pace_maker;
    mat.pace_maker_period = material_config.pace_maker_period;
    mat.signal_value_decay = material_config.signal_value_decay;
    mat.signal_time_delay = material_config.signal_time_delay;
    mat.inactive_period = material_config.inactive_period;

    // Voxel material attributes
    mat.nom_size = lattice_dim;

    ictx.voxel_materials.emplace_back(mat);
}

Vindex VX3_VoxelyzeKernelManager::addOrGetLinkMaterial(Vindex voxel1_material_index,
                                                       Vindex voxel2_material_index) {
    auto link_material_index = voxel_materials_to_link_material_index.find(
        {voxel1_material_index, voxel2_material_index});
    if (link_material_index != voxel_materials_to_link_material_index.end())
        return link_material_index->second;
    VX3_LinkMaterial link_material;
    link_material.vox1_mat = voxel1_material_index;
    link_material.vox2_mat = voxel2_material_index;
    Vindex new_link_material_index = ictx.link_materials.size();
    voxel_materials_to_link_material_index[{
        voxel1_material_index, voxel2_material_index}] = new_link_material_index;

    ictx.link_materials.emplace_back(link_material);
    return new_link_material_index;
}

void VX3_VoxelyzeKernelManager::addVoxels(const VX3_StructureConfig &structure_config,
                                          Vfloat vox_size) {
    for (int z = 0; z < structure_config.z_voxels; z++) {
        for (int y = 0; y < structure_config.y_voxels; y++) {
            for (int x = 0; x < structure_config.x_voxels; x++) {
                int idx_1d = index3DToIndex1D(x, y, z, structure_config.x_voxels,
                                              structure_config.y_voxels);
                // Skip empty voxels
                if (structure_config.data[idx_1d] == 0)
                    continue;
                auto voxel = VX3_Voxel();
                voxel.index_x = (short)x;
                voxel.index_y = (short)y;
                voxel.index_z = (short)z;
                voxel.position.x = (Vfloat)x * vox_size;
                voxel.position.y = (Vfloat)y * vox_size;
                voxel.position.z = (Vfloat)z * vox_size;
                voxel.initial_position = voxel.position;
                voxel.voxel_material = (Vindex)(structure_config.data[idx_1d]);
                voxel.amplitude = structure_config.amplitudes[idx_1d];
                voxel.frequency = structure_config.frequencies[idx_1d];
                voxel.phase_offset = structure_config.phase_offsets[idx_1d];
                voxel.base_cilia_force = structure_config.base_cilia_force[idx_1d];
                voxel.shift_cilia_force = structure_config.shift_cilia_force[idx_1d];
                for (unsigned int &link : voxel.links) {
                    link = NULL_INDEX;
                }
                coordinate_to_voxel_index[index3DToCoordinate(x, y, z)] =
                    ictx.voxels.size();
                ictx.voxels.emplace_back(voxel);
                // add any possible links utilizing this voxel
                for (int i = 0; i < 6; i++) {
                    addLink(x, y, z, (LinkDirection)i);
                }
            }
        }
    }
}

void VX3_VoxelyzeKernelManager::addLink(int x, int y, int z, LinkDirection direction) {
    auto voxel1_coords = index3DToCoordinate(x, y, z);
    auto voxel2_coords = index3DToCoordinate(x + xIndexVoxelOffset(direction),
                                             y + yIndexVoxelOffset(direction),
                                             z + zIndexVoxelOffset(direction));
    if (coordinate_to_voxel_index.find(voxel1_coords) ==
            coordinate_to_voxel_index.end() ||
        coordinate_to_voxel_index.find(voxel2_coords) == coordinate_to_voxel_index.end())
        return;

    // Since a link can only be added when voxel2 pre-exists, and
    // voxel2 cannot create a link before voxel1 is added, there is
    // no need to check duplicate links because all links are unique
    VX3_Link link;
    link.axis = linkDirectionToAxis(direction);
    // Make sure that link is always pointed to +X, +Y and +Z direction
    // from negative voxel to positive voxel
    bool reverse_order = false;
    if (link.axis == X_AXIS)
        reverse_order = direction == X_NEG;
    else if (link.axis == Y_AXIS)
        reverse_order = direction == Y_NEG;
    else if (link.axis == Z_AXIS)
        reverse_order = direction == Z_NEG;

    if (reverse_order) {
        link.voxel_neg = coordinate_to_voxel_index[voxel2_coords];
        link.voxel_pos = coordinate_to_voxel_index[voxel1_coords];
    } else {
        link.voxel_neg = coordinate_to_voxel_index[voxel1_coords];
        link.voxel_pos = coordinate_to_voxel_index[voxel2_coords];
    }
    link.link_material = addOrGetLinkMaterial(ictx.voxels[link.voxel_neg].voxel_material,
                                              ictx.voxels[link.voxel_pos].voxel_material);

    Vindex new_link_index = ictx.links.size();
    ictx.links.emplace_back(link);

    // Store link reference to voxels
    ictx.voxels[link.voxel_neg].links[direction] = new_link_index;
    ictx.voxels[link.voxel_pos].links[oppositeLinkDirection(direction)] = new_link_index;
}

void VX3_VoxelyzeKernelManager::setMathExpression(
    VX3_MathTreeToken *tokens, const VX3_Config::VX3_MathTreeExpression &expr) {
    if (expr.size() > MAX_EXPRESSION_TOKENS)
        throw std::invalid_argument("Math expression size too large");
    for (size_t i = 0; i < expr.size(); i++)
        tokens[i] = expr[i];
}

inline string VX3_VoxelyzeKernelManager::index3DToCoordinate(int x, int y, int z) const {
    return (format{"%d,%d,%d"} % x % y % z).str();
}

inline int VX3_VoxelyzeKernelManager::index3DToIndex1D(int x, int y, int z, int x_size,
                                                       int y_size) const {
    return z * (x_size * y_size) + y * x_size + x;
}