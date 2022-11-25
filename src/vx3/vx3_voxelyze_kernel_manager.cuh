#ifndef VX3_VOXELYZE_KERNEL_MANAGER_H
#define VX3_VOXELYZE_KERNEL_MANAGER_H

#include "utils/vx3_def.h"
#include "utils/vx3_hash.h"
#include "vx3/vx3_context.h"
#include "vx3/vx3_link.h"
#include "vx3/vx3_link_material.h"
#include "vx3/vx3_voxel.h"
#include "vx3/vx3_voxel_material.h"
#include "vx3/vx3_voxelyze_kernel.cuh"
#include "vxa/vx3_config.h"
#include <unordered_map>
#include <vector>

class VX3_VoxelyzeKernelManager {
  public:
    VX3_InitContext ictx;

    VX3_VoxelyzeKernel createKernelFromConfig(const VX3_Config &config,
                                              const cudaStream_t &stream);
    void freeKernel(VX3_VoxelyzeKernel &kernel);

  private:
    std::unordered_map<std::string, Vindex> coordinate_to_voxel_index;
    std::unordered_map<UnorderedPair<Vindex>, Vindex, UnorderedPairHash<Vindex>>
        voxel_materials_to_link_material_index;

    void addVoxelMaterial(const VX3_PaletteMaterialConfig &material_config,
                          int material_id, Vfloat lattice_dim, Vfloat internal_damping,
                          Vfloat collision_damping, Vfloat global_damping);
    void addVoxels(const VX3_StructureConfig &structure_config, Vfloat vox_size);
    Vindex addOrGetLinkMaterial(Vindex voxel1_material_index,
                                Vindex voxel2_material_index);
    void addLink(int x, int y, int z, LinkDirection direction);
    void setMathExpression(VX3_MathTreeToken *tokens,
                           const VX3_Config::VX3_MathTreeExpression &expr);

    // the voxel X index offset of a voxel across a link in the specified direction
    int xIndexVoxelOffset(LinkDirection direction) const {
        return (direction == X_NEG) ? -1 : ((direction == X_POS) ? 1 : 0);
    }

    // the voxel Y index offset of a voxel across a link in the specified direction
    int yIndexVoxelOffset(LinkDirection direction) const {
        return (direction == Y_NEG) ? -1 : ((direction == Y_POS) ? 1 : 0);
    }

    // the voxel Z index offset of a voxel across a link in the specified direction
    int zIndexVoxelOffset(LinkDirection direction) const {
        return (direction == Z_NEG) ? -1 : ((direction == Z_POS) ? 1 : 0);
    }

    std::string index3DToCoordinate(int x, int y, int z) const;
    int index3DToIndex1D(int x, int y, int z, int x_size, int y_size) const;
};

#endif // VX3_VOXELYZE_KERNEL_MANAGER_H