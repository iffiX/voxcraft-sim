//
// Created by iffi on 11/20/22.
//

#ifndef VX3_SIMULATION_RECORD_H
#define VX3_SIMULATION_RECORD_H
#include <vector>
#include "utils/vx3_def.h"
#include "utils/vx3_vec3d.h"

struct VX3_SimulationVoxelRecord {
    bool valid = false;
    Vindex material = NULL_INDEX;
    Vfloat local_signal = 0;
    Vfloat orient_angle = 0, orient_x = 0, orient_y = 0, orient_z = 0;
    Vfloat x = 0, y = 0, z = 0;
    Vfloat ppp_x = 0, ppp_y = 0, ppp_z = 0;
    Vfloat nnn_x = 0, nnn_y = 0, nnn_z = 0;
};
struct VX3_SimulationLinkRecord {
    bool valid = false;
    Vfloat pos_x = 0, pos_y = 0, pos_z = 0;
    Vfloat neg_x = 0, neg_y = 0, neg_z = 0;
};

struct VX3_SimulationRecord {
    using VoxelMaterialRecord = std::tuple<int, Vfloat, Vfloat, Vfloat, Vfloat>;
    Vfloat rescale;
    Vfloat vox_size;
    Vfloat dt_frac;
    int real_step_size;
    Vfloat recommended_time_step;

    std::vector<VoxelMaterialRecord> voxel_materials;
    std::vector<int> steps;
    std::vector<Vfloat> time_points;
    // When choose to not record voxels/links,
    // the sub vector will be empty
    std::vector<std::vector<VX3_SimulationVoxelRecord>> voxel_frames;
    std::vector<std::vector<VX3_SimulationLinkRecord>> link_frames;
};

struct VX3_SimulationResult {
    Vfloat current_time = 0.0;
    Vfloat fitness_score;
    Vec3f initial_center_of_mass;
    Vec3f current_center_of_mass;
    int num_close_pairs = 0;

    int num_voxel;
    int num_measured_voxel = 0;
    Vfloat vox_size;
    // sum of euclidean distances of all voxels (end - init)
    Vfloat total_distance_of_all_voxels = 0.0;
    bool save_position_of_all_voxels = false;

    std::vector<int> voxel_materials;
    std::vector<Vec3f> voxel_initial_positions;
    std::vector<Vec3f> voxel_final_positions;
};

std::string saveSimulationResult(const std::string &input_dir,
                          const std::string &vxa_filename, const std::string &vxd_filename,
                          const VX3_SimulationResult &result);

void saveSimulationResultToFile(const std::string &output_path, const std::string &input_dir,
                          const std::string &vxa_filename, const std::string &vxd_filename,
                          const VX3_SimulationResult &result);

std::string saveSimulationRecord(const VX3_SimulationRecord &record);

void saveSimulationRecordToFile(const std::string &output_path, const VX3_SimulationRecord &record);
#endif //VX3_SIMULATION_RECORD_H
