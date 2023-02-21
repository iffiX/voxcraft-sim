//
// Created by iffi on 11/19/22.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "vx3/vx3.hpp"

namespace py = pybind11;
using namespace std;

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