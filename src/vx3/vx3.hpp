#ifndef VX3_HPP
#define VX3_HPP

#include "utils/vx3_barrier.h"
#include "utils/vx3_conf.h"
#include "vx3/vx3_simulation_manager.h"
#include "vxa/vx3_config.h"
#include <future>
#include <iostream>
#include <tuple>
#include <vector>

class Voxcraft {
  public:
    using Result = std::tuple<std::vector<std::string>, std::vector<std::string>>;
    using RawResult =
        std::tuple<std::vector<VX3_SimulationResult>, std::vector<VX3_SimulationRecord>>;
    using SubResult =
        std::tuple<std::vector<VX3_SimulationResult>, std::vector<VX3_SimulationRecord>>;
    using SaveResult = std::pair<std::string, std::string>;
    std::vector<int> devices;
    size_t batch_size_per_device;

    explicit Voxcraft(const std::vector<int> &devices_ = {},
                      size_t _batch_size_per_device = 1024)
        : batch_size_per_device(_batch_size_per_device) {
        if (devices_.empty()) {
            int count;
            VcudaGetDeviceCount(&count);
            std::cout << "Devices not specified, using all " << count << " GPUs."
                      << std::endl;
            for (int i = 0; i < count; i++)
                devices.push_back(i);
        }
    }

    Result runSims(const std::vector<std::string> &base_configs,
                   const std::vector<std::string> &experiment_configs,
                   bool save_result = true, bool save_record = true) {
        Result results;
        RawResult raw_results;

        if (base_configs.size() != experiment_configs.size())
            throw std::invalid_argument(
                "Base config num is different from experiment config num.");
        if (base_configs.empty())
            return std::move(results);
        size_t batch_size = devices.size() * batch_size_per_device;
        size_t batches = CEIL(base_configs.size(), batch_size);
        size_t offset = 0;
        for (size_t b = 0; b < batches; b++) {
            size_t experiment_num = MIN(batch_size, base_configs.size() - offset);
            size_t experiments_per_device = CEIL(experiment_num, devices.size());

            std::vector<std::future<SubResult>> async_raw_results;

            for (size_t t = 0; t < experiment_num; t += experiments_per_device) {
                int device = devices[t / experiments_per_device];
                size_t start = offset;
                size_t end = start + experiments_per_device;
                std::vector<std::string> sub_batch_base_configs(
                    base_configs.begin() + start, base_configs.begin() + end);
                std::vector<std::string> sub_batch_experiment_configs(
                    experiment_configs.begin() + start, experiment_configs.begin() + end);
                async_raw_results.emplace_back(async(
                    &Voxcraft::runBatchedSims, sub_batch_base_configs,
                    sub_batch_experiment_configs, save_result, save_record, device, b));
            }
            for (auto &result : async_raw_results) {
                auto experiment_result = result.get();
                std::get<0>(raw_results)
                    .insert(std::get<0>(raw_results).end(),
                            std::get<0>(experiment_result).begin(),
                            std::get<0>(experiment_result).end());
                std::get<1>(raw_results)
                    .insert(std::get<1>(raw_results).end(),
                            std::get<1>(experiment_result).begin(),
                            std::get<1>(experiment_result).end());
            }
            offset += experiment_num;
        }

        std::vector<std::future<SaveResult>> async_results;
        for (size_t i = 0; i < base_configs.size(); i++) {
            async_results.emplace_back(std::async(&Voxcraft::saveResultAndRecord,
                                                  std::get<0>(raw_results)[i],
                                                  std::get<1>(raw_results)[i]));
        }
        for (size_t i = 0; i < base_configs.size(); i++) {
            auto res = async_results[i].get();
            std::get<0>(results).emplace_back(res.first);
            std::get<1>(results).emplace_back(res.second);
        }
        return std::move(results);
    }

  private:
    static SubResult runBatchedSims(const std::vector<std::string> &base_configs,
                                    const std::vector<std::string> &experiment_configs,
                                    bool save_result, bool save_record, int device,
                                    int batch) {
        VX3_SimulationManager sm(device, batch);
        sm.init();
        for (size_t i = 0; i < base_configs.size(); i++) {
            auto config = VX3_Config(base_configs[i], experiment_configs[i]);
            sm.addSim(config);
        }

        sm.runSims(1000000, save_result, save_record);
        sm.free();
        SubResult result;
        for (auto &sim : sm.sims) {
            std::get<0>(result).emplace_back(sim.result);
            std::get<1>(result).emplace_back(sim.record);
        }
        return std::move(result);
    }

    static SaveResult saveResultAndRecord(const VX3_SimulationResult &result,
                                          const VX3_SimulationRecord &record) {
        return make_pair(saveSimulationResult("", "", "", result),
                         saveSimulationRecord(record));
    }
};

#endif