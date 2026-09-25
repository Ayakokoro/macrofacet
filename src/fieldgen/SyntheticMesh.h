#pragma once

#include "fieldgen/PlyReader.h"

namespace mf {

// A closed lat/long sphere, centred on the origin. Exists so the whole bake
// pipeline can be exercised, and checked against a surface whose distance field
// is known analytically, without shipping a .ply asset or a mesh library.
//
// It is watertight, which matters: the winding-number sign and the unsigned-
// distance band pruning are both only exact for a closed mesh.
TriangleMesh makeSphereMesh(double radius, int segments);

} // namespace mf
