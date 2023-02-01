//
// Created by iffi on 11/23/22.
//

#include "vx3/vx3_simulation_manager.h"
#include "vxa/vx3_config.h"
#include <future>
#include <vector>

#include "utils/vx3_barrier.h"
using namespace std;

class Voxcraft {
  public:
    using Result = tuple<vector<string>, vector<string>>;
    vector<int> devices;
    size_t threads_per_device = 8;

    explicit Voxcraft(const vector<int> &devices_ = {}, size_t threads_per_device = 16)
        : threads_per_device(threads_per_device) {
        if (devices_.empty()) {
            int count;
            VcudaGetDeviceCount(&count);
            cout << "Devices not specified, using all " << count << " GPUs." << endl;
            for (int i = 0; i < count; i++)
                devices.push_back(i);
        }
    }

    Result runSims(const vector<string> &base_configs,
                   const vector<string> &experiment_configs) {
        Result results;
        if (base_configs.size() != experiment_configs.size())
            throw invalid_argument(
                "Base config num is different from experiment config num.");
        if (base_configs.empty())
            return std::move(results);
        size_t batch_size = threads_per_device * devices.size();
        size_t batches = CEIL(base_configs.size(), batch_size);
        size_t offset = 0;
        for (size_t b = 0; b < batches; b++) {
            size_t experiment_num = MIN(batch_size, base_configs.size() - offset);
            size_t experiments_per_device = CEIL(experiment_num, devices.size());
            vector<future<tuple<string, string>>> async_results;
            // auto barrier = new VX3_Barrier(experiment_num);
            VX3_Barrier *barrier = nullptr;
            for (size_t t = 0; t < experiment_num; t++) {
                int device = devices[t / experiments_per_device];
                async_results.emplace_back(
                    async(&Voxcraft::runSim, base_configs[offset + t],
                          experiment_configs[offset + t], barrier, device));
            }
            for (auto &result : async_results) {
                auto experiment_result = result.get();
                get<0>(results).emplace_back(get<0>(experiment_result));
                get<1>(results).emplace_back(get<1>(experiment_result));
            }
            // delete barrier;
            offset += experiment_num;
        }
        return std::move(results);
    }

  private:
    static tuple<string, string> runSim(const string &base_config,
                                        const string &experiment_config, VX3_Barrier *b,
                                        int device) {
        VX3_SimulationManager sm;
        auto config = VX3_Config(base_config, experiment_config);
        sm.initSim(config, device);
        if (b != nullptr)
            b->wait();
        sm.runSim();
        return std::move(make_tuple(saveSimulationResult("", "", "", sm.getResult()),
                                    saveSimulationRecord(sm.getRecord())));
    }
};

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
            ifstream base_file(base_config_path);
            ifstream robot_file(file.path().string());
            if (not base_file.is_open() || not robot_file.is_open())
                throw std::invalid_argument("Invalid files");
            stringstream base_buffer, robot_buffer;
            base_buffer << base_file.rdbuf();
            robot_buffer << robot_file.rdbuf();
            vector<string> bases, robots;
            string base = base_buffer.str();
            string robot = robot_buffer.str();
            for (size_t i = 0; i < 1; i++) {
                bases.push_back(base);
                robots.push_back(robot);
            }
            Voxcraft vx({}, 16);
            vx.runSims(bases, robots);
        }
    }
}