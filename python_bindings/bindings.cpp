#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <sstream>

#include <autograd/scalar.hpp>

namespace py = pybind11;
using autograd::Scalar;
using autograd::ScalarPtr;

PYBIND11_MODULE(pyautograd, m) {
  py::class_<Scalar, std::shared_ptr<Scalar>>(m, "Scalar")
      .def(py::init<double>())
      .def(py::init<int>())
      .def_property("data", &Scalar::data, &Scalar::set_data)
      .def_property("grad", &Scalar::grad, &Scalar::set_grad)
      .def("backward", &Scalar::backward)
      .def("__neg__", [](ScalarPtr lhs) { return -lhs; })
      .def("__add__", [](ScalarPtr lhs, ScalarPtr rhs) { return lhs + rhs; })
      .def("__add__", [](ScalarPtr lhs, double rhs) { return lhs + rhs; })
      .def("__radd__", [](ScalarPtr lhs, ScalarPtr rhs) { return lhs + rhs; })
      .def("__radd__", [](ScalarPtr lhs, double rhs) { return lhs + rhs; })
      .def("__sub__", [](ScalarPtr lhs, ScalarPtr rhs) { return lhs - rhs; })
      .def("__rsub__", [](ScalarPtr lhs, ScalarPtr rhs) { return rhs - lhs; })
      .def("__mul__", [](ScalarPtr lhs, ScalarPtr rhs) { return lhs * rhs; })
      .def("__mul__", [](ScalarPtr lhs, double rhs) { return lhs * rhs; })
      .def("__rmul__", [](ScalarPtr lhs, ScalarPtr rhs) { return lhs * rhs; })
      .def("__rmul__", [](ScalarPtr lhs, double rhs) { return lhs * rhs; })
      .def("__truediv__", [](ScalarPtr lhs, ScalarPtr rhs) { return lhs / rhs; })
      .def("__truediv__", [](ScalarPtr lhs, double rhs) { return lhs / rhs; })
      .def("__rtruediv__", [](ScalarPtr lhs, ScalarPtr rhs) { return rhs / lhs; })
      .def("__rtruediv__", [](ScalarPtr lhs, double rhs) { return rhs / lhs; })
      .def("__pow__", [](ScalarPtr lhs, ScalarPtr rhs) { return lhs->pow(rhs); })
      .def("__pow__", [](ScalarPtr lhs, double rhs) { return lhs->pow(rhs); })
      .def("__repr__", [](const Scalar& val) {
        std::stringstream ss;
        ss << val;
        return ss.str();
      });

}

