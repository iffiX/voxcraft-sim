#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include "vx3/vx3_simulation_manager.h"
#include "vxa/vx3_config.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace std;

#define APP_DESCRIPTION                                                                  \
    "\
Thank you for using Voxelyze3. This program should be run on a computer that has GPUs.\n\
Typical Usage:\n\
voxcraft-sim -i <data_path> -o <report_path> \n\n\
Allowed options\
"

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
            sm.initSim(config, 0);
            if (sm.runSim()) {
                // Use the same file name for saving
                string output_result_path =
                    (output / (file.path().stem().string() + ".result")).string();
                string output_record_path =
                    (output / (file.path().stem().string() + ".history")).string();
                saveSimulationRecordToFile(output_record_path, sm.getRecord());
                saveSimulationResultToFile(output_result_path, input.string(), "base.vxa",
                                           file.path().filename().string(),
                                           sm.getResult());
            }
        }
    }
}