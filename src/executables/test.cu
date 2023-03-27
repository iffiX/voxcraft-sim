//
// Created by iffi on 11/23/22.
//

#include "vx3/vx3.hpp"
#include <fmt/format.h>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <future>
#include <iostream>
#include <vector>

using namespace std;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

#define APP_DESCRIPTION                                                                  \
    "\
Thank you for using Voxelyze3. This program should be run on a computer that has GPUs.\n\
Typical Usage:\n\
voxcraft-sim -i <data_path> -o <report_path> \n\n\
Allowed options\
"

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
//    for (auto &file : fs::directory_iterator(input)) {
//        auto base_config_path = (input / "base.vxa").string();
//        if (boost::algorithm::to_lower_copy(file.path().extension().string()) == ".vxd") {
//            ifstream base_file(base_config_path);
//            ifstream robot_file(file.path().string());
//            stringstream base_buffer, robot_buffer;
//            base_buffer << base_file.rdbuf();
//            robot_buffer << robot_file.rdbuf();
//            auto base = base_buffer.str();
//            auto robot = robot_buffer.str();
//
//            vector<string> bases, robots;
//            for (size_t i = 0; i < 256; i++) {
//                bases.push_back(base);
//                robots.push_back(robot);
//            }
//            Voxcraft vx({}, 128);
//            auto result = vx.runSims(bases, robots);
//
//            ofstream result_file((output / "sim.result").string());
//            ofstream record_file((output / "sim.history").string());
//            if (not result_file.is_open() or not record_file.is_open())
//                throw std::invalid_argument("Cannot open output path");
//            result_file << get<0>(result)[0];
//            record_file << get<1>(result)[0];
//        }
//    }

    auto base_config_path = (input / "base.vxa").string();
    ifstream base_file(base_config_path);
    stringstream base_buffer;
    base_buffer << base_file.rdbuf();
    auto base = base_buffer.str();

    vector<string> bases, robots;

    for (size_t i = 0; i < 128; i++) {
        auto robot_config_path = (input / fmt::format("{}.vxd", i)).string();
        ifstream robot_file(robot_config_path);
        stringstream robot_buffer;
        robot_buffer << robot_file.rdbuf();
        auto robot = robot_buffer.str();
//        bases.push_back(base);
//        robots.push_back(robot);
        VX3_Config config(base, robot);
    }

//    Voxcraft vx({}, 128);
//    auto result = vx.runSims(bases, robots);
}