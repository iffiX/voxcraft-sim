#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>

#include "vx3/vx3_simulation_manager.h"
#include "vxa/vx3_config.h"

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;
namespace po = boost::program_options;
using namespace std;

#define APP_DESCRIPTION                                                                  \
    "\
Thank you for using Voxelyze3. This program should be run on a computer that has GPUs.\n\
Typical Usage:\n\
voxcraft-sim -i <data_path> -o <report_path> \n\n\
Allowed options\
"

vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void saveSimulationResult(const string &output_path, const string &input_dir,
                          const string &vxa_filename, const string &vxd_filename,
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
    pt::write_xml(output_path, tr_result);
}

void saveSimulationRecord(const string &output_path, const VX3_SimulationRecord &record) {
    ofstream file(output_path);
    stringstream ss;

    if (not file.is_open())
        throw std::invalid_argument("Cannot open output path");

    // rescale the whole space. so history file can contain less digits. ( e.g. not
    // 0.000221, but 2.21 )
    ss << "{{{setting}}}<rescale>0.001</rescale>" << endl;

    // materials color
    for (auto &mat : record.voxel_materials) {
        ss << boost::format{"{{{setting}}}"
                            "<matcolor>"
                            "<id>%d</id>"
                            "<r>%.2f</r>"
                            "<g>%.2f</g>"
                            "<b>%.2f</b>"
                            "<a>%.2f</a>"
                            "</matcolor>"} %
                  get<0>(mat) % get<1>(mat) % get<2>(mat) % get<3>(mat) % get<4>(mat)
           << endl;
    }
    ss << "{{{setting}}}<voxel_size>" << record.vox_size << "</voxel_size>" << endl;
    ss << boost::format{"real_stepsize: %d; recommendedTimeStep %f; d_v3->DtFrac %f;"} %
              record.real_step_size % record.recommended_time_step % record.dt_frac
       << endl;

    for (size_t f = 0; f < record.steps.size(); f++) {
        if (not record.voxel_frames[f].empty()) {
            ss << boost::format{"<<<Step%d Time:%f>>>"} % record.steps[f] %
                      record.time_points[f]
               << endl;
            for (auto &v : record.voxel_frames[f]) {
                ss << boost::format{"%.1f,%.1f,%.1f,"} % v.x % v.y % v.z;
                ss << boost::format{"%.1f,%.2f,%.2f,%.2f,"} % v.orient_angle %
                          v.orient_x % v.orient_y % v.orient_z;
                ss << boost::format{"%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,"} % v.nnn_x %
                          v.nnn_y % v.nnn_z % v.ppp_x % v.ppp_y % v.ppp_z;
                ss << boost::format{"%d,"} % v.material;       // for coloring
                ss << boost::format{"%.1f,"} % v.local_signal; // for coloring as well.
                ss << ";";
            }
            ss << "<<<>>>";
        }
        if (not record.link_frames[f].empty()) {
            // Links
            ss << boost::format{"|[[[%d]]]"} % record.steps[f];
            for (auto &l : record.link_frames[f]) {
                ss << boost::format{"%.4f,%.4f,%.4f,"} % l.pos_x % l.pos_y % l.pos_z;
                ss << boost::format{"%.4f,%.4f,%.4f,"} % l.neg_x % l.neg_y % l.neg_z;
                ss << ";";
            }
            ss << "[[[]]]";
        }
        ss << endl;
    }
    file << ss.rdbuf();
}

//<data_path> should contain a file named `base.vxa' and multiple files with extension `.vxa'.\n\
//<report_path> is the report file you need to place. If you want to overwrite existing report, add -f flag.\n\
//if the executable `vx3_node_worker' doesn't exist in the same path, use -w <worker> to specify the path.\n\

int main(int argc, char **argv) {
    // setup tools for parsing arguments
    po::options_description desc(APP_DESCRIPTION);
    desc.add_options()("help,h", "produce help message")(
        "input,i", po::value<string>(),
        "Set input directory path which contains a generation of vxa files.")(
        "output,o", po::value<string>(),
        "Set output directory path for history and report. (e.g. report/");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // check parameters
    if (vm.count("help") || !vm.count("input") || !vm.count("output")) {
        cout << desc << "\n";
        return 1;
    }
    fs::path input(vm["input"].as<string>());
    fs::path output(vm["output"].as<string>());
    if (!fs::is_directory(input)) {
        cout << "Error: input directory not found.\n\n";
        cout << desc << "\n";
        return 1;
    }
    if (!fs::is_directory(output)) {
        fs::create_directories(output);
    }
    if (!fs::is_regular_file(input / "base.vxa")) {
        cout << "No base.vxa found in input directory.\n\n";
        cout << desc << "\n";
        return 1;
    }

    // count number of GPUs
    int nDevices = 0;
    VcudaGetDeviceCount(&nDevices);
    if (nDevices <= 0) {
        printf(COLORCODE_BOLD_RED "ERROR: No GPU found.\n");
        return 1;
    } else {
        printf("%d GPU found.\n", nDevices);
    }

    // Currently, just use GPU 0
    for (auto &file : fs::directory_iterator(input)) {
        auto base_config_path = (input / "base.vxa").string();
        if (boost::algorithm::to_lower_copy(file.path().extension().string()) == ".vxd") {
            auto config = VX3_Config();
            config.open(base_config_path, file.path().string());
            cout << "Running simulation for file: " << file.path().string() << endl;
            VX3_SimulationManager sm;
            if (sm.runSim(config, 0)) {
                // Use the same file name for saving
                string output_result_path =
                    (output / (file.path().stem().string() + ".result")).string();
                string output_record_path =
                    (output / (file.path().stem().string() + ".history")).string();
                saveSimulationRecord(output_record_path, sm.getRecord());
                saveSimulationResult(output_result_path, input.string(), "base.vxa",
                                     file.path().filename().string(), sm.getResult());
            }
        }
    }
}