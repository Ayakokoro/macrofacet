# igl_shim — a one-file workaround for a vcpkg port defect

## What this is

`raytri.c` (Tomas Möller's ray-triangle intersection code, **public domain**).

## Why it is here

The vcpkg `libigl` port installs only `*.h` under `installed/x64-windows/include/igl/`.
But `igl/AABB.h` reaches this C file by a chain that ends in a *quoted* include:

```
igl/AABB.h → AABB.cpp → ray_mesh_intersect.h → ray_mesh_intersect.cpp
           → extern "C" { #include "raytri.c" }
```

so any translation unit that includes `<igl/AABB.h>` fails with
`fatal error C1083: cannot open include file: 'raytri.c'`.

This directory is placed on the include path (`igl_raytri_shim`, see
`CMakeLists.txt`). Because the include is quoted, MSVC searches the includer's own
directory first and then falls back to the include path — so the shim is found.

## Provenance

Copied verbatim (267 lines, unmodified) from the libigl v2.6.0 source tree that the
vcpkg port itself builds from:

```
<vcpkg>/buildtrees/libigl/src/v2.6.0-*/include/igl/raytri.c
```

Keep it byte-identical to upstream so it can be diffed when the port is upgraded.
The file's own header states the license ("Alec: this file is listed as
'Public Domain'"), so vendoring it carries no MPL-2.0 obligation — the rest of libigl
remains MPL-2.0 under its own installed license.

## Exit condition

If a future `libigl` port starts installing `raytri.c`, delete this directory, the
`igl_raytri_shim` target, and the link of that target.
