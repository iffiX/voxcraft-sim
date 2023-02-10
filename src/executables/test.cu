//
// Created by iffi on 11/23/22.
//

#include "utils/vx3_barrier.h"
#include "utils/vx3_conf.h"
#include "vx3/vx3_simulation_manager.h"
#include "vxa/vx3_config.h"
#include <future>
#include <vector>

using namespace std;

class Voxcraft {
public:
    using Result = tuple<vector<string>, vector<string>>;
    using SubResult = tuple<vector<string>, vector<string>>;
    vector<int> devices;
    size_t threads_per_device = 4;
    size_t sub_batch_size_per_thread = 4;

    explicit Voxcraft(const vector<int> &devices_ = {}, size_t threads_per_device = 4,
                      size_t sub_batch_size_per_thread = 4)
            : threads_per_device(threads_per_device),
              sub_batch_size_per_thread(sub_batch_size_per_thread) {

        if (sub_batch_size_per_thread > VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE) {
            cout << "Sub batch size exceed max allowed size "
                 << VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE << endl;
            cout << "Adjusting it to max allowed size" << endl;
            this->sub_batch_size_per_thread = VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE;
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
                   const vector<string> &experiment_configs,
                   bool barrier_on_init = true) {
        Result results;
        if (base_configs.size() != experiment_configs.size())
            throw invalid_argument(
                    "Base config num is different from experiment config num.");
        if (base_configs.empty())
            return std::move(results);
        size_t batch_size =
                devices.size() * threads_per_device * sub_batch_size_per_thread;
        size_t batches = CEIL(base_configs.size(), batch_size);
        size_t offset = 0;
        for (size_t b = 0; b < batches; b++) {
            size_t experiment_num = MIN(batch_size, base_configs.size() - offset);
            size_t experiments_per_device = CEIL(experiment_num, devices.size());
            size_t experiments_per_thread =
                    CEIL(experiments_per_device, threads_per_device);

            vector<future<SubResult>> async_results;

            VX3_Barrier *barrier = nullptr;
            if (barrier_on_init)
                barrier = new VX3_Barrier(CEIL(experiment_num, experiments_per_thread));

            for (size_t t = 0; t < experiment_num; t += experiments_per_thread) {
                int device = devices[t / experiments_per_device];
                size_t experiment_num_in_thread =
                        MIN(experiments_per_thread, experiment_num - t);
                size_t start = offset + t;
                size_t end = start + experiment_num_in_thread;
                vector<string> sub_batch_base_configs(base_configs.begin() + start,
                                                      base_configs.begin() + end);
                vector<string> sub_batch_experiment_configs(
                        experiment_configs.begin() + start, experiment_configs.begin() + end);
                async_results.emplace_back(
                        async(&Voxcraft::runBatchedSims, sub_batch_base_configs,
                              sub_batch_experiment_configs, barrier, device));
            }
            for (auto &result : async_results) {
                auto experiment_result = result.get();
                get<0>(results).insert(get<0>(results).end(),
                                       get<0>(experiment_result).begin(),
                                       get<0>(experiment_result).end());
                get<1>(results).insert(get<1>(results).end(),
                                       get<1>(experiment_result).begin(),
                                       get<1>(experiment_result).end());
            }
            if (barrier_on_init)
                delete barrier;
            offset += experiment_num;
        }
        return std::move(results);
    }

private:
    static SubResult runBatchedSims(const vector<string> &base_configs,
                                    const vector<string> &experiment_configs,
                                    VX3_Barrier *b, int device) {
        VX3_SimulationManager sm(device);
        for (size_t i = 0; i < base_configs.size(); i++) {
            auto config = VX3_Config(base_configs[i], experiment_configs[i]);
            sm.addSim(config);
        }
        if (b != nullptr)
            b->wait();
        sm.runSims();
        SubResult result;
        for (auto &sim : sm.sims) {
            get<0>(result).emplace_back(saveSimulationResult("", "", "", sim.result));
            get<1>(result).emplace_back(saveSimulationRecord(sim.record));
        }
        return std::move(result);
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
//            stringstream base_buffer, robot_buffer;
//            base_buffer << base_file.rdbuf();
//            robot_buffer << robot_file.rdbuf();
//            auto base = base_buffer.str();
//            auto robot = robot_buffer.str();

            std::string base( (std::istreambuf_iterator<char>(base_file) ),
                                 (std::istreambuf_iterator<char>()    ) );
            std::string robot( (std::istreambuf_iterator<char>(robot_file) ),
                              (std::istreambuf_iterator<char>()    ) );

            vector<string> bases, robots;
            for (size_t i = 0; i < 32; i++) {
                bases.push_back(base);
                robots.push_back(robot);
            }
            Voxcraft vx({}, 1, 18);
            vx.runSims(bases, robots);
        }
    }
}