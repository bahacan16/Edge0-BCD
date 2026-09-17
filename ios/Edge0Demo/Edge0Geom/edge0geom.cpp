// Polygon offsetting and boolean union, as a built-in Python module.
//
// This exists because `shapely` cannot run here and never will: it is a
// wrapper around GEOS, which is C++, and iOS does not load compiled Python
// extensions that were not built into the app. But the part of shapely the
// DXF→PIM script actually needs is small — `buffer` for cutter-radius
// compensation, `union` for repairing self-intersections — and those two are
// Clipper2, which is two headers and three source files with no dependencies.
//
// Built in rather than shipped as a `.so`. CPython on iOS can load an
// extension from a framework (that is what the 67 stdlib modules do), but a
// module registered with `PyImport_AppendInittab` before `Py_Initialize`
// needs no framework, no code signature and no `.fwork` placeholder — it is
// simply part of the app binary. For a module we compile ourselves that is
// strictly less machinery.
//
// The geometry that is *not* here is deliberate. Area, bounds, centroid,
// point-in-polygon and the affine transforms are twenty lines of Python each
// and belong in the shim, where they can be read and corrected. Only the two
// operations that are genuinely hard — offsetting with round joins, and
// resolving the self-intersections that offsetting creates — cross into C++.

#define PY_SSIZE_T_CLEAN
#include <Python/Python.h>

#include <exception>
#include <string>
#include <vector>

#include "clipper2/clipper.h"

namespace {

using Clipper2Lib::PathD;
using Clipper2Lib::PathsD;
using Clipper2Lib::PointD;

/// A ring is a sequence of (x, y) pairs; a value is a sequence of rings.
/// Tuples or lists, because the shim builds one and the caller may pass the
/// other.
bool pathsFromPython(PyObject *object, PathsD &out) {
    PyObject *rings = PySequence_Fast(object, "expected a sequence of rings");
    if (rings == nullptr) return false;
    const Py_ssize_t ringCount = PySequence_Fast_GET_SIZE(rings);
    out.clear();
    out.reserve(static_cast<size_t>(ringCount));

    for (Py_ssize_t i = 0; i < ringCount; ++i) {
        PyObject *ring = PySequence_Fast_GET_ITEM(rings, i);
        PyObject *points = PySequence_Fast(ring, "expected a sequence of points");
        if (points == nullptr) {
            Py_DECREF(rings);
            return false;
        }
        const Py_ssize_t pointCount = PySequence_Fast_GET_SIZE(points);
        PathD path;
        path.reserve(static_cast<size_t>(pointCount));

        for (Py_ssize_t j = 0; j < pointCount; ++j) {
            PyObject *point = PySequence_Fast_GET_ITEM(points, j);
            PyObject *pair = PySequence_Fast(point, "expected an (x, y) pair");
            if (pair == nullptr) {
                Py_DECREF(points);
                Py_DECREF(rings);
                return false;
            }
            if (PySequence_Fast_GET_SIZE(pair) < 2) {
                PyErr_SetString(PyExc_ValueError, "a point needs two coordinates");
                Py_DECREF(pair);
                Py_DECREF(points);
                Py_DECREF(rings);
                return false;
            }
            const double x = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(pair, 0));
            const double y = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(pair, 1));
            Py_DECREF(pair);
            if (PyErr_Occurred()) {
                Py_DECREF(points);
                Py_DECREF(rings);
                return false;
            }
            path.push_back(PointD(x, y));
        }
        Py_DECREF(points);
        out.push_back(std::move(path));
    }
    Py_DECREF(rings);
    return true;
}

PyObject *pythonFromPaths(const PathsD &paths) {
    PyObject *rings = PyList_New(static_cast<Py_ssize_t>(paths.size()));
    if (rings == nullptr) return nullptr;

    for (size_t i = 0; i < paths.size(); ++i) {
        const PathD &path = paths[i];
        PyObject *points = PyList_New(static_cast<Py_ssize_t>(path.size()));
        if (points == nullptr) {
            Py_DECREF(rings);
            return nullptr;
        }
        for (size_t j = 0; j < path.size(); ++j) {
            PyObject *pair = Py_BuildValue("(dd)", path[j].x, path[j].y);
            if (pair == nullptr) {
                Py_DECREF(points);
                Py_DECREF(rings);
                return nullptr;
            }
            PyList_SET_ITEM(points, static_cast<Py_ssize_t>(j), pair);
        }
        PyList_SET_ITEM(rings, static_cast<Py_ssize_t>(i), points);
    }
    return rings;
}

Clipper2Lib::JoinType joinTypeNamed(const char *name) {
    const std::string value(name == nullptr ? "round" : name);
    if (value == "miter" || value == "mitre") return Clipper2Lib::JoinType::Miter;
    if (value == "bevel") return Clipper2Lib::JoinType::Bevel;
    if (value == "square") return Clipper2Lib::JoinType::Square;
    return Clipper2Lib::JoinType::Round;
}

Clipper2Lib::FillRule fillRuleNamed(const char *name) {
    const std::string value(name == nullptr ? "nonzero" : name);
    if (value == "evenodd") return Clipper2Lib::FillRule::EvenOdd;
    if (value == "positive") return Clipper2Lib::FillRule::Positive;
    if (value == "negative") return Clipper2Lib::FillRule::Negative;
    return Clipper2Lib::FillRule::NonZero;
}

PyObject *geomOffset(PyObject *, PyObject *args, PyObject *keywords) {
    PyObject *input = nullptr;
    double delta = 0.0;
    const char *join = "round";
    double miterLimit = 2.0;
    int precision = 4;
    double arcTolerance = 0.0;

    static const char *names[] = {
        "paths", "delta", "join", "miter_limit", "precision", "arc_tolerance", nullptr};
    // The cast is what the C API asks for: it takes a mutable char** it does
    // not write to.
    if (!PyArg_ParseTupleAndKeywords(
            args, keywords, "Od|sdid", const_cast<char **>(names), &input, &delta, &join,
            &miterLimit, &precision, &arcTolerance)) {
        return nullptr;
    }

    PathsD subjects;
    if (!pathsFromPython(input, subjects)) return nullptr;

    try {
        // EndType::Polygon: closed rings offset outwards for a positive delta
        // and inwards for a negative one, which is the cutter-radius
        // convention the script is written against.
        const PathsD result = Clipper2Lib::InflatePaths(
            subjects, delta, joinTypeNamed(join), Clipper2Lib::EndType::Polygon, miterLimit,
            precision, arcTolerance);
        return pythonFromPaths(result);
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_ValueError, "clipper2 offset failed");
        return nullptr;
    }
}

PyObject *geomUnion(PyObject *, PyObject *args, PyObject *keywords) {
    PyObject *input = nullptr;
    PyObject *other = Py_None;
    const char *fill = "nonzero";
    int precision = 4;

    static const char *names[] = {"paths", "other", "fill", "precision", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, keywords, "O|Osi", const_cast<char **>(names), &input, &other, &fill,
            &precision)) {
        return nullptr;
    }

    PathsD subjects;
    if (!pathsFromPython(input, subjects)) return nullptr;

    try {
        const Clipper2Lib::FillRule rule = fillRuleNamed(fill);
        if (other == Py_None) {
            // A self-union is how a ring that crosses itself is made valid —
            // this is what `shapely`'s `buffer(0)` idiom is for.
            return pythonFromPaths(Clipper2Lib::Union(subjects, rule, precision));
        }
        PathsD clips;
        if (!pathsFromPython(other, clips)) return nullptr;
        return pythonFromPaths(Clipper2Lib::Union(subjects, clips, rule, precision));
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_ValueError, "clipper2 union failed");
        return nullptr;
    }
}

PyObject *geomVersion(PyObject *, PyObject *) {
    return PyUnicode_FromString(CLIPPER2_VERSION);
}

PyMethodDef methods[] = {
    {"offset", reinterpret_cast<PyCFunction>(geomOffset), METH_VARARGS | METH_KEYWORDS,
     "offset(paths, delta, join='round', miter_limit=2.0, precision=4, arc_tolerance=0.0)"},
    {"union", reinterpret_cast<PyCFunction>(geomUnion), METH_VARARGS | METH_KEYWORDS,
     "union(paths, other=None, fill='nonzero', precision=4)"},
    {"version", geomVersion, METH_NOARGS, "The vendored Clipper2 version."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef moduleDefinition = {
    PyModuleDef_HEAD_INIT,
    "_edge0geom",
    "Polygon offsetting and union, backed by Clipper2.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyObject *initialiseModule(void) { return PyModule_Create(&moduleDefinition); }

}  // namespace

extern "C" void edge0_register_geom_module(void) {
    // Must happen before Py_Initialize: the inittab is read once, while the
    // import machinery is being built.
    PyImport_AppendInittab("_edge0geom", initialiseModule);
}
