#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "gibbs.hpp"
#include "gibbs_multisensor.hpp"
#include "ms_adaptive_birth/gm_adaptive_birth.hpp"
#include "ms_adaptive_birth/mc_adaptive_birth.hpp"
#include "ms_glmb_ukf/ms_glmb_ukf.hpp"

namespace py = pybind11;

PYBIND11_MODULE(lrfscpp, m) {
    m.def("gibbs_jointpredupdt", &gibbs_jointpredupdt);
    m.def("gibbs_multisensor_approx_cheap", &gibbs_multisensor_approx_cheap);
    m.def("gibbs_multisensor_approx_dprobsample", &gibbs_multisensor_approx_dprobsample);

    // Multi-sensor Joint Adaptive Birth Sampler
    py::class_<AdaptiveBirth>(m, "AdaptiveBirth")
        .def(py::init<>())
        .def(py::init<int>())
        .def("sample_adaptive_birth", &AdaptiveBirth::sample_adaptive_birth)
        .def("init_parameters", &AdaptiveBirth::init_parameters);
    py::class_<MCAdaptiveBirth>(m, "MCAdaptiveBirth")
        .def(py::init<>())
        .def(py::init<vector<MatrixXd>, int, int, double, double, double>())
        .def("sample_adaptive_birth", &MCAdaptiveBirth::sample_adaptive_birth);

    // Multi-sensor GLMB UKF
    py::class_<MSGLMB>(m, "MSGLMB")
        .def(py::init<vector<MatrixXd>>())
        .def("run_msglmb_ukf", &MSGLMB::run_msglmb_ukf); // .def_readwrite("m", &Target::m)
}
