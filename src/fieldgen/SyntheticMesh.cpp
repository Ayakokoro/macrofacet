#include "fieldgen/SyntheticMesh.h"

#include "macrofacet/core/Types.h"

#include <algorithm>
#include <cmath>

namespace mf {

TriangleMesh makeSphereMesh(double radius, int segments) {
    const int nu = std::max(8, segments);   // around the equator
    const int nv = std::max(4, nu / 2);     // pole to pole

    TriangleMesh mesh;
    mesh.vertices.resize((nu + 1) * (nv + 1), 3);
    for (int j = 0; j <= nv; ++j) {
        const double phi = kPi * j / nv;
        for (int i = 0; i <= nu; ++i) {
            const double theta = 2.0 * kPi * i / nu;
            mesh.vertices.row(j * (nu + 1) + i)
                << radius * std::sin(phi) * std::cos(theta),
                   radius * std::sin(phi) * std::sin(theta), radius * std::cos(phi);
        }
    }

    // The seam is welded (column nu reuses column 0's position as its own
    // vertex) rather than indexed into column 0: the winding number and the
    // point-triangle distance both work per triangle, so a shared position is
    // enough and no index remapping is needed.
    mesh.faces.resize(nu * nv * 2, 3);
    int face = 0;
    for (int j = 0; j < nv; ++j) {
        for (int i = 0; i < nu; ++i) {
            const int a = j * (nu + 1) + i;
            const int b = a + 1;
            const int c = a + nu + 1;
            const int d = c + 1;
            mesh.faces.row(face++) << a, c, b;
            mesh.faces.row(face++) << b, c, d;
        }
    }
    return mesh;
}

} // namespace mf
