//
// Created by iffi on 11/5/22.
//

#include "vxa/vx3_config.h"

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>
#include <iostream>
#include <queue>
#include <stack>

using namespace std;
using namespace boost;
using namespace fmt;
namespace pt = boost::property_tree;

void split(const string &s, char delim, vector<string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        // In case of trailing delimiter
        if (item.find_first_not_of(' ') != std::string::npos)
            elems.push_back(item);
    }
}

VX3_Config::VX3_Config(const string &base_config_str, const string &config_str) {
    base_config_stream << base_config_str;
    config_stream << config_str;
    if (base_config_str.empty() or config_str.empty())
        return;
    parseSettings();
}

void VX3_Config::open(const string &base_config_path, const string &config_path) {
    ifstream base_config_file(base_config_path);
    ifstream config_file(config_path);
    if (!base_config_file.is_open())
        throw std::invalid_argument("Invalid base config file.");
    if (!config_file.is_open())
        throw std::invalid_argument("Invalid config file.");
    base_config_stream << base_config_file.rdbuf();
    config_stream << config_file.rdbuf();
    parseSettings();
}

void VX3_Config::parseSettings() {
    // The base VXA file and per simulation VXD file
    pt::ptree pt_baseVXA, pt_VXD;
    pt::read_xml(base_config_stream, pt_baseVXA);
    pt::read_xml(config_stream, pt_VXD);
    merge(pt_baseVXA, pt_VXD);
    config_tree = pt_baseVXA;
    lattice.read(config_tree.get_child("VXA.VXC.Lattice"));
    palette.read(config_tree.get_child("VXA.VXC.Palette"));
    structure.read(config_tree.get_child("VXA.VXC.Structure"));

    // Validate structure
    for (auto d : structure.data) {
        if (d < 0 || d >= palette.materials.size()) {
            throw std::runtime_error(fmt::format(
                "Material {:d} referenced in structure data doesn't exist", d));
        }
    }

    if (not structure.has_phase_offsets) {
        for (size_t i = 0; i < structure.phase_offsets.size(); i++) {
            structure.phase_offsets[i] =
                palette.materials[structure.data[i]].material_temp_phase;
        }
    }

    // VXA.RawPrint
    raw_print = config_tree.get<std::string>("VXA.RawPrint", "");
    if (not raw_print.empty())
        std::cout << raw_print << std::endl;

    // In VXA.GPU
    heap_size = max(min(config_tree.get("VXA.GPU.HeapSize", (Vfloat)0.5), (Vfloat)0.99),
                    (Vfloat)0.01);

    // In VXA.Simulator.Integration
    dt_frac = config_tree.get("VXA.Simulator.Integration.DtFrac", (Vfloat)0.9);

    // In VXA.Simulator.Damping
    bond_damping_z = config_tree.get("VXA.Simulator.Damping.BondDampingZ", (Vfloat)0.1);
    col_damping_z = config_tree.get("VXA.Simulator.Damping.ColDampingZ", (Vfloat)1.0);
    slow_damping_z = config_tree.get("VXA.Simulator.Damping.SlowDampingZ", (Vfloat)1.0);

    // In VXA.Simulator.StopCondition
    parseMathExpression(stop_condition_formula,
                        config_tree.get_child_optional(
                            "VXA.Simulator.StopCondition.StopConditionFormula"));

    // In VXA.Simulator.RecordHistory
    record_step_size = config_tree.get("VXA.Simulator.RecordHistory.RecordStepSize", 0);
    record_link = config_tree.get("VXA.Simulator.RecordHistory.RecordLink", 0);
    record_voxel = config_tree.get("VXA.Simulator.RecordHistory.RecordVoxel", 1);

    // In VXA.Simulator.AttachDetach
    enable_collision =
        config_tree.get("VXA.Simulator.AttachDetach.EnableCollision", false);
    enable_attach = config_tree.get("VXA.Simulator.AttachDetach.EnableAttach", false);
    enable_detach = config_tree.get("VXA.Simulator.AttachDetach.EnableDetach", false);
    watch_distance =
        config_tree.get("VXA.Simulator.AttachDetach.watchDistance", (Vfloat)1.0);
    bounding_radius =
        config_tree.get("VXA.Simulator.AttachDetach.boundingRadius", (Vfloat)0.75);
    safety_guard = config_tree.get("VXA.Simulator.AttachDetach.SafetyGuard", 500);
    attach_conditions.resize(5);
    parseMathExpression(attach_conditions[0],
                        config_tree.get_child_optional(
                            "VXA.Simulator.AttachDetach.AttachCondition.Condition_0"));
    parseMathExpression(attach_conditions[1],
                        config_tree.get_child_optional(
                            "VXA.Simulator.AttachDetach.AttachCondition.Condition_1"));
    parseMathExpression(attach_conditions[2],
                        config_tree.get_child_optional(
                            "VXA.Simulator.AttachDetach.AttachCondition.Condition_2"));
    parseMathExpression(attach_conditions[3],
                        config_tree.get_child_optional(
                            "VXA.Simulator.AttachDetach.AttachCondition.Condition_3"));
    parseMathExpression(attach_conditions[4],
                        config_tree.get_child_optional(
                            "VXA.Simulator.AttachDetach.AttachCondition.Condition_4"));

    // In VXA.Simulator.ForceField
    parseMathExpression(x_force_field, config_tree.get_child_optional(
                                           "VXA.Simulator.ForceField.x_force_field"));
    parseMathExpression(y_force_field, config_tree.get_child_optional(
                                           "VXA.Simulator.ForceField.y_force_field"));
    parseMathExpression(z_force_field, config_tree.get_child_optional(
                                           "VXA.Simulator.ForceField.z_force_field"));

    // In VXA.Simulator
    parseMathExpression(fitness_function,
                        config_tree.get_child_optional("VXA.Simulator.FitnessFunction"));
    save_position_of_all_voxels =
        config_tree.get("VXA.Simulator.SavePositionOfAllVoxels", 0);
    max_dist_in_voxel_lengths_to_count_as_pair =
        config_tree.get("VXA.Simulator.MaxDistInVoxelLengthsToCountAsPair", (Vfloat)0);

    enable_cilia = config_tree.get("VXA.Simulator.EnableCilia", 0);
    enable_signals = config_tree.get("VXA.Simulator.EnableSignals", 0);

    // In VXA.Environment.Gravity
    grav_enabled = config_tree.get("VXA.Environment.Gravity.GravEnabled", 0); // ?
    floor_enabled = config_tree.get("VXA.Environment.Gravity.FloorEnabled", 0);
    grav_acc = config_tree.get("VXA.Environment.Gravity.GravAcc", (Vfloat)-9.81);

    // In VXA.Environment.Thermal
    enable_vary_temp = config_tree.get("VXA.Environment.Thermal.VaryTempEnabled", 0);
    temp_amplitude = config_tree.get("VXA.Environment.Thermal.TempAmplitude", (Vfloat)0);
    temp_period = config_tree.get("VXA.Environment.Thermal.TempPeriod", (Vfloat)0.1);
}

void VX3_Config::parseMathExpression(VX3_MathTreeExpression &expr,
                                     const boost::optional<pt::ptree &> &expr_tree) {
    if (not expr_tree)
        return;
    vector<pair<string, string>> raw_tokens;
    postFixTraversal(expr_tree.get(), "mtEND", "", raw_tokens);

    // pop from stack to expression (so we get a reversed order)
    for (auto &tok : raw_tokens) {
        VX3_MathTreeToken token;
        if (tok.first == "mtEND") {
            token.op = mtEND;
        } else if (tok.first == "mtVAR") {
            token.op = mtVAR;
            if (tok.second == "x") {
                token.value = 0;
            } else if (tok.second == "y") {
                token.value = 1;
            } else if (tok.second == "z") {
                token.value = 2;
            } else if (tok.second == "hit") {
                token.value = 3;
            } else if (tok.second == "t") {
                token.value = 4;
            } else if (tok.second == "angle") {
                token.value = 5;
            } else if (tok.second == "targetCloseness") {
                token.value = 6;
            } else if (tok.second == "numClosePairs") {
                token.value = 7;
            } else if (tok.second == "num_voxel") {
                token.value = 8;
            } else {
                throw std::invalid_argument(
                    format("Unknown token variable: {}", tok.second));
            }
        } else if (tok.first == "mtCONST") {
            token.op = mtCONST;
            try {
                token.value = (Vfloat)std::stod(tok.second);
            } catch (...) {
                throw std::invalid_argument("Using token mtCONST with no number.");
            }
        } else if (tok.first == "mtADD") {
            token.op = mtADD;
        } else if (tok.first == "mtSUB") {
            token.op = mtSUB;
        } else if (tok.first == "mtMUL") {
            token.op = mtMUL;
        } else if (tok.first == "mtDIV") {
            token.op = mtDIV;
        } else if (tok.first == "mtPOW") {
            token.op = mtPOW;
        } else if (tok.first == "mtSQRT") {
            token.op = mtSQRT;
        } else if (tok.first == "mtE") {
            token.op = mtE;
        } else if (tok.first == "mtPI") {
            token.op = mtPI;
        } else if (tok.first == "mtSIN") {
            token.op = mtSIN;
        } else if (tok.first == "mtCOS") {
            token.op = mtCOS;
        } else if (tok.first == "mtTAN") {
            token.op = mtTAN;
        } else if (tok.first == "mtATAN") {
            token.op = mtATAN;
        } else if (tok.first == "mtLOG") {
            token.op = mtLOG;
        } else if (tok.first == "mtINT") {
            token.op = mtINT;
        } else if (tok.first == "mtABS") {
            token.op = mtABS;
        } else if (tok.first == "mtNOT") {
            token.op = mtNOT;
        } else if (tok.first == "mtGREATERTHAN") {
            token.op = mtGREATERTHAN;
        } else if (tok.first == "mtLESSTHAN") {
            token.op = mtLESSTHAN;
        } else if (tok.first == "mtAND") {
            token.op = mtAND;
        } else if (tok.first == "mtOR") {
            token.op = mtOR;
        } else if (tok.first == "mtNORMALCDF") {
            token.op = mtNORMALCDF;
        } else {
            throw std::invalid_argument(format("Unknown token operation: {}", tok.first));
        }
        expr.push_back(token);
    }
    if (not VX3_MathTree::isExpressionValid(expr.data()))
        throw std::invalid_argument("Expression is invalid");
}

void VX3_Config::postFixTraversal(const pt::ptree &expr_tree, const std::string &root_op,
                                  const std::string &root_value,
                                  vector<pair<string, string>> &raw_tokens) {
    for (auto &child : expr_tree.get_child("")) {
        string value = child.second.data();
        trim_right(value);
        string op = child.first;
        trim_right(op);
        postFixTraversal(child.second, op, value, raw_tokens);
    }
    raw_tokens.emplace_back(root_op, root_value);
}

void VX3_Config::merge(pt::ptree &vxa, const pt::ptree &vxd) {
    for (auto &child : vxd.get_child("VXD", vxd).get_child("")) {
        std::string replace = child.second.get<std::string>("<xmlattr>.replace", "");
        if (replace.length() > 0) {
            vxa.put_child(replace, child.second);
        }
    }
}

void VX3_LatticeConfig::read(const boost::property_tree::ptree &lattice_tree) {
    // root of tree: VXC.Lattice
    lattice_dim = lattice_tree.get("Lattice_Dim", (Vfloat)0.001);
}

void VX3_PaletteConfig::read(const boost::property_tree::ptree &palette_tree) {
    // root of tree: VXC.Palette
    auto default_material = VX3_PaletteMaterialConfig();
    default_material.material_id = 0;
    materials.emplace_back(default_material);
    int material_id = 1;
    for (auto &child : palette_tree.get_child("")) {
        auto material = VX3_PaletteMaterialConfig();
        auto id = child.second.get<string>("<xmlattr>.ID", "");
        if (not id.empty()) {
            if (stoi(id) != material_id)
                throw std::runtime_error("ID is not consecutive");
        }
        material.read(child.second);
        material.material_id = material_id;
        materials.emplace_back(material);
        material_id++;
    }
}

void VX3_PaletteMaterialConfig::read(
    const boost::property_tree::ptree &palette_material_tree) {
    // root of tree: VXC.Palette.Material[N]
    name = palette_material_tree.get("Name", "Default");

    r = palette_material_tree.get("Display.Red", (Vfloat)0.5);
    g = palette_material_tree.get("Display.Green", (Vfloat)0.5);
    b = palette_material_tree.get("Display.Blue", (Vfloat)0.5);
    a = palette_material_tree.get("Display.Alpha", (Vfloat)1.0);

    // TODO: Unify XML names
    is_target = palette_material_tree.get("Mechanical.isTarget", false);
    is_measured = palette_material_tree.get("Mechanical.isMeasured", true);
    fixed = palette_material_tree.get("Mechanical.Fixed", false);
    sticky = palette_material_tree.get("Mechanical.Sticky", false);
    cilia = palette_material_tree.get("Mechanical.Cilia", (Vfloat)0.0);
    is_pace_maker = palette_material_tree.get("Mechanical.isPaceMaker", false);
    pace_maker_period =
        palette_material_tree.get("Mechanical.PaceMakerPeriod", (Vfloat)0.0);
    signal_value_decay =
        palette_material_tree.get("Mechanical.signalValueDecay", (Vfloat)0.9);
    signal_time_delay =
        palette_material_tree.get("Mechanical.signalTimeDelay", (Vfloat)0.0);
    inactive_period = palette_material_tree.get("Mechanical.inactivePeriod", (Vfloat)0.0);
    mat_model =
        palette_material_tree.get<int>("Mechanical.MatModel", MaterialModel::MAT_LINEAR);

    elastic_mod = palette_material_tree.get("Mechanical.Elastic_Mod", (Vfloat)0.0);
    if (mat_model == MaterialModel::MAT_LINEAR_FAIL)
        fail_stress = palette_material_tree.get("Mechanical.Fail_Stress", (Vfloat)0.0);
    else
        fail_stress = -1;
    density = palette_material_tree.get("Mechanical.Density", (Vfloat)0.0);
    poissons_ratio = palette_material_tree.get("Mechanical.Poissons_Ratio", (Vfloat)0.0);
    CTE = palette_material_tree.get("Mechanical.CTE", (Vfloat)0.0);
    u_static = palette_material_tree.get("Mechanical.uStatic", (Vfloat)0.0);
    u_dynamic = palette_material_tree.get("Mechanical.uDynamic", (Vfloat)0.0);
    material_temp_phase =
        palette_material_tree.get("Mechanical.MaterialTempPhase", (Vfloat)0.0);
}

void VX3_StructureConfig::read(const boost::property_tree::ptree &structure_tree) {
    // root of tree: VXC.Structure
    // Requires size to present, no default values
    if (structure_tree.get<string>("<xmlattr>.Compression") != "ASCII_READABLE")
        throw std::invalid_argument(
            "Only ASCII_READABLE compression mode is supported now.");

    x_voxels = structure_tree.get<int>("X_Voxels");
    y_voxels = structure_tree.get<int>("Y_Voxels");
    z_voxels = structure_tree.get<int>("Z_Voxels");

    size_t voxel_num = x_voxels * y_voxels * z_voxels;
    size_t voxel_per_layer_num = x_voxels * y_voxels;
    data.resize(voxel_num, 0);
    // When global amplitude is 0, amplitudes here are invalidated
    amplitudes.resize(voxel_num, 1);
    frequencies.resize(voxel_num, 1);
    phase_offsets.resize(voxel_num, 0);
    base_cilia_force.resize(voxel_num);
    shift_cilia_force.resize(voxel_num);

    size_t l = 0;
    for (auto &layer : structure_tree.get_child("Data")) {
        auto raw_layer = layer.second.get<string>("");
        if (raw_layer.length() != voxel_per_layer_num)
            throw std::invalid_argument(format("Data layer {} size is {}, required {}", l,
                                               raw_layer.length(), voxel_per_layer_num));
        for (size_t i = 0; i < voxel_per_layer_num; i++)
            data[l * voxel_per_layer_num + i] = (char)(raw_layer[i] - '0');
        l++;
    }
    l = 0;
    has_amplitudes =
        read_float_layer(structure_tree, "Amplitude", voxel_per_layer_num, amplitudes);
    has_frequencies =
        read_float_layer(structure_tree, "Frequency", voxel_per_layer_num, frequencies);
    has_phase_offsets = read_float_layer(structure_tree, "PhaseOffset",
                                         voxel_per_layer_num, phase_offsets);
    has_base_cilia_force = read_vec3f_layer(structure_tree, "BaseCiliaForce",
                                            voxel_per_layer_num, base_cilia_force);
    has_shift_cilia_force = read_vec3f_layer(structure_tree, "ShiftCiliaForce",
                                             voxel_per_layer_num, shift_cilia_force);
}

bool VX3_StructureConfig::read_float_layer(
    const boost::property_tree::ptree &structure_tree, const std::string &section,
    size_t voxel_per_layer_num, std::vector<Vfloat> &layer_data) {
    if (not structure_tree.get_child_optional(section))
        return false;
    size_t l = 0;
    for (auto &layer : structure_tree.get_child(section)) {
        vector<string> splitted_values;
        split(layer.second.get<string>(""), ',', splitted_values);
        if (splitted_values.size() != voxel_per_layer_num)
            throw std::invalid_argument(format("{} layer {} size is {}, required {}",
                                               section, l, splitted_values.size(),
                                               voxel_per_layer_num));
        for (size_t i = 0; i < voxel_per_layer_num; i++) {
            Vfloat val = stod(splitted_values[i]);
            if (not isfinite(val))
                throw std::invalid_argument(
                    format("{} layer {} has invalid float value: {}", section, l,
                           splitted_values[i]));
            layer_data[l * voxel_per_layer_num + i] = val;
        }
        l++;
    }
    return true;
}

bool VX3_StructureConfig::read_vec3f_layer(
    const boost::property_tree::ptree &structure_tree, const std::string &section,
    size_t voxel_per_layer_num, std::vector<Vec3f> &layer_data) {
    if (not structure_tree.get_child_optional(section))
        return false;
    size_t l = 0;
    for (auto &layer : structure_tree.get_child(section)) {
        vector<string> splitted_values;
        split(layer.second.get<string>(""), ',', splitted_values);
        if (splitted_values.size() != 3 * voxel_per_layer_num)
            throw std::invalid_argument(format("{} layer {} size is {}, required {}",
                                               section, l, splitted_values.size(),
                                               3 * voxel_per_layer_num));
        for (size_t i = 0; i < voxel_per_layer_num; i++) {
            Vfloat val0 = stod(splitted_values[i * 3]),
                   val1 = stod(splitted_values[i * 3 + 1]),
                   val2 = stod(splitted_values[i * 3 + 2]);
            if (not isfinite(val0))
                throw std::invalid_argument(
                    format("{} layer {} has invalid Vec3f value at position 0: {}",
                           section, l, splitted_values[i * 3]));
            if (not isfinite(val1))
                throw std::invalid_argument(
                    format("{} layer {} has invalid Vec3f value at position 1: {}",
                           section, l, splitted_values[i * 3 + 1]));
            if (not isfinite(val2))
                throw std::invalid_argument(
                    format("{} layer {} has invalid Vec3f value at position 2: {}",
                           section, l, splitted_values[i * 3 + 2]));
            layer_data[l * voxel_per_layer_num + i] = Vec3f(val0, val1, val2);
        }
        l++;
    }
    return true;
}