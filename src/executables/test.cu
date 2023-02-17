//
// Created by iffi on 11/23/22.
//

#include <future>
#include <vector>
#include <backward-cpp/backward.hpp>

#include "utils/vx3_barrier.h"
#include "utils/vx3_conf.h"
#include "vx3/vx3_simulation_manager.h"
#include "vxa/vx3_config.h"

using namespace std;

class Voxcraft {
  public:
    using Result = tuple<vector<string>, vector<string>>;
    using RawResult = tuple<vector<VX3_SimulationResult>, vector<VX3_SimulationRecord>>;
    using SubResult = tuple<vector<VX3_SimulationResult>, vector<VX3_SimulationRecord>>;
    using SaveResult = pair<string, string>;
    vector<int> devices;
    size_t batch_size_per_device;

    explicit Voxcraft(const vector<int> &devices_ = {},
                      size_t _batch_size_per_device = VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE)
        : batch_size_per_device(_batch_size_per_device) {
        if (batch_size_per_device > VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE) {
            cout << "Batch size per device exceeds limit "
                 << VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE << ", scaling down to accommodate"
                 << endl;
            batch_size_per_device = VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE;
        }
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
        RawResult raw_results;

        if (base_configs.size() != experiment_configs.size())
            throw invalid_argument(
                "Base config num is different from experiment config num.");
        if (base_configs.empty())
            return std::move(results);
        size_t batch_size = devices.size() * batch_size_per_device;
        size_t batches = CEIL(base_configs.size(), batch_size);
        size_t offset = 0;
        for (size_t b = 0; b < batches; b++) {
            size_t experiment_num = MIN(batch_size, base_configs.size() - offset);
            size_t experiments_per_device = CEIL(experiment_num, devices.size());

            vector<future<SubResult>> async_raw_results;

            for (size_t t = 0; t < experiment_num; t += experiments_per_device) {
                int device = devices[t / experiments_per_device];
                size_t start = offset;
                size_t end = start + experiments_per_device;
                vector<string> sub_batch_base_configs(base_configs.begin() + start,
                                                      base_configs.begin() + end);
                vector<string> sub_batch_experiment_configs(
                    experiment_configs.begin() + start, experiment_configs.begin() + end);
                async_raw_results.emplace_back(
                    async(&Voxcraft::runBatchedSims, sub_batch_base_configs,
                          sub_batch_experiment_configs, device, b));
            }
            for (auto &result : async_raw_results) {
                auto experiment_result = result.get();
                get<0>(raw_results)
                    .insert(get<0>(raw_results).end(), get<0>(experiment_result).begin(),
                            get<0>(experiment_result).end());
                get<1>(raw_results)
                    .insert(get<1>(raw_results).end(), get<1>(experiment_result).begin(),
                            get<1>(experiment_result).end());
            }
            offset += experiment_num;
        }

        vector<future<SaveResult>> async_results;
        for (size_t i = 0; i < base_configs.size(); i++) {
            async_results.emplace_back(async(&Voxcraft::saveResultAndRecord,
                                             get<0>(raw_results)[i],
                                             get<1>(raw_results)[i]));
        }
        for (size_t i = 0; i < base_configs.size(); i++) {
            auto res = async_results[i].get();
            get<0>(results).emplace_back(res.first);
            get<1>(results).emplace_back(res.second);
        }
        return std::move(results);
    }

  private:
    static SubResult runBatchedSims(const vector<string> &base_configs,
                                    const vector<string> &experiment_configs,
                                    int device,
                                    int batch) {
        try {
            VX3_SimulationManager sm(device, batch);
            for (size_t i = 0; i < base_configs.size(); i++) {
                auto config = VX3_Config(base_configs[i], experiment_configs[i]);
                sm.addSim(config);
            }
            sm.runSims();
            SubResult result;
            for (auto &sim: sm.sims) {
                get<0>(result).emplace_back(sim.result);
                get<1>(result).emplace_back(sim.record);
            }

            return std::move(result);
        } catch (const std::exception &exc) {
            cerr << "Exception on device " << device << " batch " << batch << endl;
            cerr << exc.what();
        }
    }

    static SaveResult saveResultAndRecord(const VX3_SimulationResult &result,
                                          const VX3_SimulationRecord &record) {
        return make_pair(saveSimulationResult("", "", "", result),
                         saveSimulationRecord(record));
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
    backward::SignalHandling sh{};
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
            //            stringstream base_buffer, robot_buffer;
            //            base_buffer << base_file.rdbuf();
            //            robot_buffer << robot_file.rdbuf();
            //            auto base = base_buffer.str();
            //            auto robot = robot_buffer.str();

            std::string base((std::istreambuf_iterator<char>(base_file)),
                             (std::istreambuf_iterator<char>()));
            std::string robot((std::istreambuf_iterator<char>(robot_file)),
                              (std::istreambuf_iterator<char>()));

            vector<string> bases, robots;
            for (size_t i = 0; i < 256; i++) {
                bases.push_back(base);
                robots.push_back(robot);
            }
            Voxcraft vx({}, 128);
            auto result = vx.runSims(bases, robots);

            ofstream result_file((output / "sim.result").string());
            ofstream record_file((output / "sim.history").string());
            if (not result_file.is_open() or not record_file.is_open())
                throw std::invalid_argument("Cannot open output path");
            result_file << get<0>(result)[0];
            record_file << get<1>(result)[0];
        }
    }
}