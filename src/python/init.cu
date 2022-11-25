//
// Created by iffi on 11/19/22.
//

#include "utils/vx3_barrier.h"
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
    vector<int> devices;
    size_t threads_per_device = 8;

    explicit Voxcraft(const vector<int> &devices_ = {}, size_t threads_per_device = 32)
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
                   const vector<string> &experiment_configs,
                   bool barrier_on_init = true) {
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

            VX3_Barrier *barrier = nullptr;
            if (barrier_on_init)
                barrier = new VX3_Barrier(experiment_num);

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
            if (barrier_on_init)
                delete barrier;
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

PYBIND11_MODULE(voxcraft, m) {
    py::class_<Voxcraft>(m, "Voxcraft")
        .def(py::init<const vector<int> &, size_t>(), py::arg("devices") = vector<int>{},
             py::arg("threads_per_device") = 32)
        .def_readwrite("devices", &Voxcraft::devices)
        .def_readwrite("threads_per_device", &Voxcraft::threads_per_device)
        .def("run_sims", &Voxcraft::runSims, py::arg("base_configs"),
             py::arg("experiment_configs"), py::arg("barrier_on_init") = true);
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}