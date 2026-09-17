// Bridged into Swift so the module can be registered before the interpreter
// starts. See edge0geom.cpp.

#ifndef EDGE0_GEOM_H
#define EDGE0_GEOM_H

#ifdef __cplusplus
extern "C" {
#endif

/// Adds `_edge0geom` to CPython's table of built-in modules. Call exactly
/// once, and before `Py_Initialize`.
void edge0_register_geom_module(void);

#ifdef __cplusplus
}
#endif

#endif  // EDGE0_GEOM_H
