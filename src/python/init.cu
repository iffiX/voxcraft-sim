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

PYBIND11_MODULE(voxcraft, m) {
    py::class_<Voxcraft>(m, "Voxcraft")
        .def(py::init<const vector<int> &, size_t, size_t>(),
             py::arg("devices") = vector<int>{},
             py::arg("threads_per_device") = 4,
             py::arg("sub_batch_size_per_thread") = 4)
        .def_readwrite("devices", &Voxcraft::devices)
        .def_readwrite("threads_per_device", &Voxcraft::threads_per_device)
        .def_readwrite("sub_batch_size_per_thread", &Voxcraft::sub_batch_size_per_thread)
        .def("run_sims", &Voxcraft::runSims, py::arg("base_configs"),
             py::arg("experiment_configs"), py::arg("barrier_on_init") = false);
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}