#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include "XYZgeomag.hpp"

static PyObject* field(PyObject*, PyObject* args)
{
    double year, x, y, z;
    if (!PyArg_ParseTuple(args, "dddd", &year, &x, &y, &z)) {
        return nullptr;
    }
    if (!std::isfinite(year) || !std::isfinite(x) ||
        !std::isfinite(y) || !std::isfinite(z)) {
        PyErr_SetString(PyExc_ValueError, "Magnetic inputs must be finite");
        return nullptr;
    }
    if (year < 2025.0 || year >= 2030.0) {
        PyErr_SetString(PyExc_ValueError, "WMM2025 requires 2025 <= decimal year < 2030");
        return nullptr;
    }
    const double radius = std::hypot(std::hypot(x, y), z);
    if (radius < 6350000.0 || radius > 7230000.0) {
        PyErr_SetString(PyExc_ValueError, "Position is outside the supported WMM radius");
        return nullptr;
    }
    const geomag::Vector position = {
        static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)
    };
    const geomag::Vector result =
        geomag::GeoMag(static_cast<float>(year), position, geomag::WMM2025);
    if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
        PyErr_SetString(PyExc_RuntimeError, "XYZgeomag returned a non-finite field");
        return nullptr;
    }
    return Py_BuildValue("(ddd)", result.x, result.y, result.z);
}

static PyMethodDef methods[] = {
    {"field", field, METH_VARARGS, "WMM2025 Cartesian field in tesla from ITRS meters."},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_xyzgeomag", nullptr, -1, methods
};

PyMODINIT_FUNC PyInit__xyzgeomag()
{
    return PyModule_Create(&module);
}
