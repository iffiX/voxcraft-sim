//
// Created by iffi on 11/19/22.
//

#include "utils/vx3_barrier.h"
#include "utils/vx3_conf.h"
#include "vx3/vx3_simulation_manager.h"
#include "vxa/vx3_config.h"
#include <future>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
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
        VX3_SimulationManager sm(device, batch);
        for (size_t i = 0; i < base_configs.size(); i++) {
            auto config = VX3_Config(base_configs[i], experiment_configs[i]);
            sm.addSim(config);
        }
        sm.runSims();
        SubResult result;
        for (auto &sim : sm.sims) {
            get<0>(result).emplace_back(sim.result);
            get<1>(result).emplace_back(sim.record);
        }
        return std::move(result);
    }

    static SaveResult saveResultAndRecord(const VX3_SimulationResult &result,
                                          const VX3_SimulationRecord &record) {
        return make_pair(saveSimulationResult("", "", "", result),
                         saveSimulationRecord(record));
    }
};

PYBIND11_MODULE(voxcraft, m) {
    py::class_<Voxcraft>(m, "Voxcraft")
        .def(py::init<const vector<int> &, size_t>(), py::arg("devices") = vector<int>{},
             py::arg("batch_size_per_device") = VX3_VOXELYZE_KERNEL_MAX_BATCH_SIZE)
        .def_readwrite("devices", &Voxcraft::devices)
        .def_readwrite("batch_size_per_device", &Voxcraft::batch_size_per_device)
        .def("run_sims", &Voxcraft::runSims, py::arg("base_configs"),
             py::arg("experiment_configs"));
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}