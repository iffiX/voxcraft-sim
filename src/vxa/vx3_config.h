//
// Created by iffi on 11/5/22.
//

#ifndef VX3_CONFIG_H
#define VX3_CONFIG_H

#include <vector>
#include <sstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "utils/vx3_def.h"
#include "utils/vx3_vec3d.h"
#include "utils/vx3_math_tree.h"

class VX3_Config;
struct VX3_LatticeConfig;
struct VX3_PaletteConfig;
struct VX3_PaletteMaterialConfig;
struct VX3_StructureConfig;

// container for information about the lattice of possible voxel locations
struct VX3_LatticeConfig {
    //!< The base lattice dimension.
    //!< The lattice dimension defines the distance between voxels in meters.
    Vfloat lattice_dim;

    void read(const boost::property_tree::ptree &lattice_tree);
};

struct VX3_PaletteConfig {
    std::vector<VX3_PaletteMaterialConfig> materials;

    void read(const boost::property_tree::ptree &palette_tree);
};

struct VX3_PaletteMaterialConfig {
    int material_id;

    enum MaterialModel : int { MAT_LINEAR, MAT_LINEAR_FAIL };

    std::string name; // material name (unused)

    // Display
    Vfloat r, g, b, a;

    // Mechanical
    bool is_target;
    bool is_measured;
    bool fixed;
    bool sticky;
    Vfloat cilia;

    bool is_pace_maker;
    Vfloat pace_maker_period; // voltage = sin(t*PaceMakerPeriod)

    Vfloat signal_value_decay; // ratio from [0,1]
    Vfloat signal_time_delay;  // in sec
    Vfloat inactive_period;    // in sec

    int mat_model;

    Vfloat elastic_mod;
    Vfloat fail_stress;
    Vfloat density;
    Vfloat poissons_ratio;
    Vfloat CTE;
    Vfloat u_static;
    Vfloat u_dynamic;

    Vfloat material_temp_phase;

    void read(const boost::property_tree::ptree &palette_material_tree);
};

struct VX3_StructureConfig {
    int x_voxels;
    int y_voxels;
    int z_voxels;

    // The main voxel array.
    // This is an array of chars; the entries correspond with the
    // material IDs, and the position in the array corresponds with
    // the position in the 3D structure, where the array is ordered:
    // starting at (x0,x0,z0), proceeding to (xn,y0,z0), next to
    // (xn,yn,z0), and on to (xn,yn,zn)
    std::vector<char> data;

    // Other arrays, they have the same size as data
    std::vector<Vfloat> amplitudes, frequencies, phase_offsets;
    std::vector<Vec3f> base_cilia_force, shift_cilia_force;

    bool has_amplitudes, has_frequencies, has_phase_offsets,
         has_base_cilia_force, has_shift_cilia_force;

    void read(const boost::property_tree::ptree &structure_tree);
    static bool
    read_float_layer(const boost::property_tree::ptree &structure_tree,
                     const std::string &section, size_t voxel_per_layer_num,
                     std::vector<Vfloat>& layer_data);
    static bool
    read_vec3f_layer(const boost::property_tree::ptree &structure_tree,
                     const std::string &section, size_t voxel_per_layer_num,
                     std::vector<Vec3f>& layer_data);
};

class VX3_Config {
public:
    using VX3_MathTreeExpression = std::vector<VX3_MathTreeToken>;
    boost::property_tree::ptree config_tree;

    /**
     * Simulation configs
     * Include:
     * VXA.GPU, VXA.Simulator, VXA.Environment, VXA.RawPrint
     */

    // VXA.RawPrint
    std::string raw_print;

    // In VXA.GPU
    Vfloat heap_size;

    // In VXA.Simulator.Integration
    Vfloat dt_frac;

    // In VXA.Simulator.Damping
    Vfloat bond_damping_z;
    Vfloat col_damping_z;
    Vfloat slow_damping_z;

    // In VXA.Simulator.StopCondition
    VX3_MathTreeExpression stop_condition_formula;

    // In VXA.Simulator.RecordHistory
    int record_step_size;
    int record_link;
    int record_voxel;

    // In VXA.Simulator.AttachDetach
    bool enable_collision;
    bool enable_attach;
    bool enable_detach; // ?
    Vfloat watch_distance; // ?
    Vfloat bounding_radius; // ?
    int safety_guard;
    std::vector<VX3_MathTreeExpression> attach_conditions;

    // In VXA.Simulator.ForceField
    VX3_MathTreeExpression x_force_field;
    VX3_MathTreeExpression y_force_field;
    VX3_MathTreeExpression z_force_field;

    // In VXA.Simulator
    VX3_MathTreeExpression fitness_function;
    Vfloat max_dist_in_voxel_lengths_to_count_as_pair;
    int save_position_of_all_voxels;
    int enable_cilia;
    int enable_signals;

    // In VXA.Environment.Gravity
    int grav_enabled;
    int floor_enabled;
    Vfloat grav_acc;

    // In VXA.Environment.Thermal
    int enable_vary_temp;
    Vfloat temp_amplitude;
    Vfloat temp_period;

    /**
     * Voxel configs
     * Include:
     * VXA.VXC
     */
    VX3_LatticeConfig lattice;
    VX3_PaletteConfig palette;
    VX3_StructureConfig structure;

    explicit VX3_Config(const std::string &base_config_str = "",
                        const std::string &config_str = "");
    void open(const std::string &base_config_path,
              const std::string &config_path);

private:
    std::stringstream base_config_stream;
    std::stringstream config_stream;

    void parseSettings();
    static void parseMathExpression(VX3_MathTreeExpression &expr,
                                    const boost::optional<boost::property_tree::ptree&>& expr_tree);
    static void postFixTraversal(const boost::property_tree::ptree& expr_tree,
                                 const std::string& root_op,
                                 const std::string& root_value,
                                 std::vector<std::pair<std::string, std::string>>& raw_tokens);
    static void merge(boost::property_tree::ptree &vxa,
                      const boost::property_tree::ptree &vxd);
};
#endif // VX3_CONFIG_H
