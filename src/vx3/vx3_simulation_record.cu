//
// Created by iffi on 11/23/22.
//
#include "vx3_simulation_record.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <fmt/format.h>

namespace pt = boost::property_tree;
using namespace std;
using namespace fmt;

string saveSimulationResult(const string &input_dir, const string &vxa_filename,
                            const string &vxd_filename,
                            const VX3_SimulationResult &result) {
    pt::ptree tr_result;

    tr_result.put("report.input_dir", input_dir);
    tr_result.put("report.best_fit.vxa_filename", vxa_filename);
    tr_result.put("report.best_fit.vxd_filename", vxd_filename);
    tr_result.put("report.best_fit.fitness_score", result.fitness_score);

    tr_result.put("report.detail.time", result.current_time);
    tr_result.put("report.detail.fitness_score", result.fitness_score);

    tr_result.put("report.detail.num_close_pairs", result.num_close_pairs);
    tr_result.put("report.detail.initial_center_of_mass.x",
                  result.initial_center_of_mass.x);
    tr_result.put("report.detail.initial_center_of_mass.y",
                  result.initial_center_of_mass.y);
    tr_result.put("report.detail.initial_center_of_mass.z",
                  result.initial_center_of_mass.z);
    tr_result.put("report.detail.current_center_of_mass.x",
                  result.current_center_of_mass.x);
    tr_result.put("report.detail.current_center_of_mass.y",
                  result.current_center_of_mass.y);
    tr_result.put("report.detail.current_center_of_mass.z",
                  result.current_center_of_mass.z);

    tr_result.put("report.detail.num_voxel", result.num_voxel);
    tr_result.put("report.detail.num_measured_voxel", result.num_measured_voxel);
    tr_result.put("report.detail.vox_size", result.vox_size);
    tr_result.put("report.detail.total_distance_of_all_voxels",
                  result.total_distance_of_all_voxels);

    if (result.save_position_of_all_voxels) {
        string str_tmp;
        for (auto mat_id : result.voxel_materials) {
            str_tmp += to_string(mat_id) + ";";
        }
        tr_result.put("report.detail.voxel_materials", str_tmp);

        str_tmp = "";
        for (auto &pos : result.voxel_initial_positions) {
            str_tmp +=
                to_string(pos.x) + "," + to_string(pos.y) + "," + to_string(pos.z) + ";";
        }
        tr_result.put("report.detail.voxel_initial_positions", str_tmp);

        str_tmp = "";
        for (auto &pos : result.voxel_final_positions) {
            str_tmp +=
                to_string(pos.x) + "," + to_string(pos.y) + "," + to_string(pos.z) + ";";
        }
        tr_result.put("report.detail.voxel_final_positions", str_tmp);
    }

    stringstream output;
    pt::write_xml(output, tr_result);
    return output.str();
}

void saveSimulationResultToFile(const string &output_path, const string &input_dir,
                                const string &vxa_filename, const string &vxd_filename,
                                const VX3_SimulationResult &result) {
    ofstream file(output_path);
    if (not file.is_open())
        throw std::invalid_argument("Cannot open output path");
    file << saveSimulationResult(input_dir, vxa_filename, vxd_filename, result);
}

string saveSimulationRecord(const VX3_SimulationRecord &record) {
    stringstream ss;

    // rescale the whole space. so history file can contain less digits. ( e.g. not
    // 0.000221, but 2.21 )
    ss << "{{{setting}}}<rescale>0.001</rescale>" << endl;

    // materials color
    for (auto &mat : record.voxel_materials) {
        if (get<0>(mat) == 0)
            continue;
        ss << "{{{setting}}}"
           << format("<matcolor>"
                     "<id>{:d}</id>"
                     "<r>{:.2f}</r>"
                     "<g>{:.2f}</g>"
                     "<b>{:.2f}</b>"
                     "<a>{:.2f}</a>"
                     "</matcolor>",
                     get<0>(mat), get<1>(mat), get<2>(mat), get<3>(mat), get<4>(mat))
           << endl;
    }
    ss << "{{{setting}}}<voxel_size>" << record.vox_size << "</voxel_size>" << endl;
    ss << format("real_stepsize: {:d}; recommendedTimeStep {:f}; d_v3->DtFrac {:f};",
                 record.real_step_size, record.recommended_time_step, record.dt_frac)
       << endl;

    for (size_t f = 0; f < record.steps.size(); f++) {
        if (not record.voxel_frames[f].empty()) {
            ss << format("<<<Step{:d} Time:{:f}>>>", record.steps[f],
                         record.time_points[f]);
            for (auto &v : record.voxel_frames[f]) {
                ss << format("{:.1f},{:.1f},{:.1f},", v.x, v.y, v.z);
                ss << format("{:.1f},{:.2f},{:.2f},{:.2f},", v.orient_angle, v.orient_x,
                             v.orient_y, v.orient_z);
                ss << format("{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},", v.nnn_x,
                             v.nnn_y, v.nnn_z, v.ppp_x, v.ppp_y, v.ppp_z);
                ss << format("{:d},", v.material);       // for coloring
                ss << format("{:.1f},", v.local_signal); // for coloring as well.
                ss << ";";
            }
            ss << "<<<>>>";
        }
        if (not record.link_frames[f].empty()) {
            // Links
            ss << format("|[[[{:d}]]]", record.steps[f]);
            for (auto &l : record.link_frames[f]) {
                ss << format("{:.4f},{:.4f},{:.4f},", l.pos_x, l.pos_y, l.pos_z);
                ss << format("{:.4f},{:.4f},{:.4f},", l.neg_x, l.neg_y, l.neg_z);
                ss << ";";
            }
            ss << "[[[]]]";
        }
        ss << endl;
    }
    return ss.str();
}

void saveSimulationRecordToFile(const string &output_path,
                                const VX3_SimulationRecord &record) {
    ofstream file(output_path);
    if (not file.is_open())
        throw std::invalid_argument("Cannot open output path");
    file << saveSimulationRecord(record);
}