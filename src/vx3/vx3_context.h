//
// Created by iffi on 11/3/22.
//

#ifndef VX3_CONTEXT_H
#define VX3_CONTEXT_H

#include "utils/vx3_conf.h"
#include "utils/vx3_soa.h"
#include "vx3/vx3_link.h"
#include "vx3/vx3_link_material.h"
#include "vx3/vx3_voxel.h"
#include "vx3/vx3_voxel_material.h"
#include <vector>

class VX3_LinkMaterial;
class VX3_VoxelMaterial;
class VX3_Link;
class VX3_Voxel;

struct VX3_InitContext {
    std::vector<VX3_LinkMaterial> link_materials;
    std::vector<VX3_VoxelMaterial> voxel_materials;
    std::vector<VX3_Link> links;
    std::vector<VX3_Voxel> voxels;
};

struct VX3_Context {
    VX3_hdStructOfArrays<VX3_LinkMaterial> link_materials;
    VX3_hdStructOfArrays<VX3_VoxelMaterial> voxel_materials;
    VX3_hdStructOfArrays<VX3_Link> links;
    VX3_hdStructOfArrays<VX3_Voxel> voxels;
};

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#ifndef DEBUG_CUDA_KERNEL
#define DEBUG_ARGS(method)
#else
#define DEBUG_ARGS(method) , #method ": @file " __FILE__ " @line " TOSTRING(__LINE__)
#endif

// See https://stackoverflow.com/questions/3046889/optional-parameters-with-c-macros
// Empty macro are placed there for safely clear replacements when argument number is
// invalid. voxel get
#define V_G_0()
#define V_G_1(property)                                                                  \
    ctx.voxels.get<Pr(VX3_Voxel, property)>(voxel DEBUG_ARGS(voxel_get))
#define V_G_2(voxel_idx, property)                                                       \
    ctx.voxels.get<Pr(VX3_Voxel, property)>(voxel_idx DEBUG_ARGS(voxel_get))
#define V_G_X(x, A, B, FUNC, ...) FUNC
#define V_G(...)                                                                         \
    V_G_X(, ##__VA_ARGS__, V_G_2(__VA_ARGS__), V_G_1(__VA_ARGS__), V_G_0(__VA_ARGS__))

#define V_S_0()
#define V_S_1()
#define V_S_2(property, value)                                                           \
    ctx.voxels.set<Pr(VX3_Voxel, property)>(voxel, value DEBUG_ARGS(voxel_set))
#define V_S_3(voxel_idx, property, value)                                                \
    ctx.voxels.set<Pr(VX3_Voxel, property)>(voxel_idx, value DEBUG_ARGS(voxel_set))
#define V_S_X(x, A, B, C, FUNC, ...) FUNC
#define V_S(...)                                                                         \
    V_S_X(, ##__VA_ARGS__, V_S_3(__VA_ARGS__), V_S_2(__VA_ARGS__), V_S_1(__VA_ARGS__),   \
          V_S_0(__VA_ARGS__))

#define L_G_0()
#define L_G_1(property) ctx.links.get<Pr(VX3_Link, property)>(link DEBUG_ARGS(link_get))
#define L_G_2(link_idx, property)                                                        \
    ctx.links.get<Pr(VX3_Link, property)>(link_idx DEBUG_ARGS(link_get))
#define L_G_X(x, A, B, FUNC, ...) FUNC
#define L_G(...)                                                                         \
    L_G_X(, ##__VA_ARGS__, L_G_2(__VA_ARGS__), L_G_1(__VA_ARGS__), L_G_0(__VA_ARGS__))

#define L_S_0()
#define L_S_1()
#define L_S_2(property, value)                                                           \
    ctx.links.set<Pr(VX3_Link, property)>(link, value DEBUG_ARGS(link_set))
#define L_S_3(link_idx, property, value)                                                 \
    ctx.links.set<Pr(VX3_Link, property)>(link_idx, value DEBUG_ARGS(link_set))
#define L_S_X(x, A, B, C, FUNC, ...) FUNC
#define L_S(...)                                                                         \
    L_S_X(, ##__VA_ARGS__, L_S_3(__VA_ARGS__), L_S_2(__VA_ARGS__), L_S_1(__VA_ARGS__),   \
          L_S_0(__VA_ARGS__))

#define LM_G_0()
#define LM_G_1(property)                                                                 \
    ctx.link_materials.get<Pr(VX3_LinkMaterial, property)>(                              \
        material DEBUG_ARGS(link_material_get))

#define LM_G_2(link_material_idx, property)                                              \
    ctx.link_materials.get<Pr(VX3_LinkMaterial, property)>(                              \
        link_material_idx DEBUG_ARGS(link_material_get))
#define LM_G_X(x, A, B, FUNC, ...) FUNC
#define LM_G(...)                                                                        \
    LM_G_X(, ##__VA_ARGS__, LM_G_2(__VA_ARGS__), LM_G_1(__VA_ARGS__), LM_G_0(__VA_ARGS__))

#define LM_S_0()
#define LM_S_1()
#define LM_S_2(property, value)                                                          \
    ctx.link_materials.set<Pr(VX3_LinkMaterial, property)>(                              \
        material, value DEBUG_ARGS(link_material_set))

#define LM_S_3(link_material_idx, property, value)                                       \
    ctx.link_materials.set<Pr(VX3_LinkMaterial, property)>(                              \
        link_material_idx, value DEBUG_ARGS(link_material_set))

#define LM_S_X(x, A, B, C, FUNC, ...) FUNC
#define LM_S(...)                                                                        \
    LM_S_X(, ##__VA_ARGS__, LM_S_3(__VA_ARGS__), LM_S_2(__VA_ARGS__),                    \
           LM_S_1(__VA_ARGS__), LM_S_0(__VA_ARGS__))

#define VM_G_0()
#define VM_G_1(property)                                                                 \
    ctx.voxel_materials.get<Pr(VX3_VoxelMaterial, property)>(                            \
        material DEBUG_ARGS(voxel_material_get))

#define VM_G_2(voxel_material_idx, property)                                             \
    ctx.voxel_materials.get<Pr(VX3_VoxelMaterial, property)>(                            \
        voxel_material_idx DEBUG_ARGS(voxel_material_get))

#define VM_G_X(x, A, B, FUNC, ...) FUNC
#define VM_G(...)                                                                        \
    VM_G_X(, ##__VA_ARGS__, VM_G_2(__VA_ARGS__), VM_G_1(__VA_ARGS__), VM_G_0(__VA_ARGS__))

#define VM_S_0()
#define VM_S_1()
#define VM_S_2(property, value)                                                          \
    ctx.voxel_materials.set<Pr(VX3_VoxelMaterial, property)>(                            \
        material, value DEBUG_ARGS(voxel_material_set))

#define VM_S_3(voxel_material_idx, property, value)                                      \
    ctx.voxel_materials.set<Pr(VX3_VoxelMaterial, property)>(                            \
        voxel_material_idx, value DEBUG_ARGS(voxel_material_set))
#define VM_S_X(x, A, B, C, FUNC, ...) FUNC
#define VM_S(...)                                                                        \
    VM_S_X(, ##__VA_ARGS__, VM_S_3(__VA_ARGS__), VM_S_2(__VA_ARGS__),                    \
           VM_S_1(__VA_ARGS__), VM_S_0(__VA_ARGS__))
#endif // VX3_CONTEXT_H
