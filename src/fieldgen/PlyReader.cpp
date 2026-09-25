#include "fieldgen/PlyReader.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mf {
namespace {

enum class Format { Ascii, BinaryLittleEndian };

enum class Scalar { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };

int scalarSize(Scalar type) {
    switch (type) {
        case Scalar::Int8: case Scalar::UInt8: return 1;
        case Scalar::Int16: case Scalar::UInt16: return 2;
        case Scalar::Int32: case Scalar::UInt32: case Scalar::Float32: return 4;
        case Scalar::Float64: return 8;
    }
    throw std::runtime_error("unreachable scalar type");
}

Scalar parseScalar(const std::string& token) {
    if (token == "char" || token == "int8") return Scalar::Int8;
    if (token == "uchar" || token == "uint8") return Scalar::UInt8;
    if (token == "short" || token == "int16") return Scalar::Int16;
    if (token == "ushort" || token == "uint16") return Scalar::UInt16;
    if (token == "int" || token == "int32") return Scalar::Int32;
    if (token == "uint" || token == "uint32") return Scalar::UInt32;
    if (token == "float" || token == "float32") return Scalar::Float32;
    if (token == "double" || token == "float64") return Scalar::Float64;
    throw std::runtime_error("unsupported PLY scalar type: " + token);
}

struct Property {
    std::string name;
    bool isList = false;
    Scalar countType = Scalar::UInt8;
    Scalar valueType = Scalar::Float32;
    int byteOffset = 0;  // fixed-stride elements only
};

struct Element {
    std::string name;
    long count = 0;
    std::vector<Property> properties;
    int stride = 0;
    bool fixedStride = true;
};

// Reads one scalar out of an in-memory record for the binary path.
double decodeScalar(const unsigned char* p, Scalar type) {
    switch (type) {
        case Scalar::Int8: { std::int8_t v; std::memcpy(&v, p, 1); return v; }
        case Scalar::UInt8: { std::uint8_t v; std::memcpy(&v, p, 1); return v; }
        case Scalar::Int16: { std::int16_t v; std::memcpy(&v, p, 2); return v; }
        case Scalar::UInt16: { std::uint16_t v; std::memcpy(&v, p, 2); return v; }
        case Scalar::Int32: { std::int32_t v; std::memcpy(&v, p, 4); return v; }
        case Scalar::UInt32: { std::uint32_t v; std::memcpy(&v, p, 4); return v; }
        case Scalar::Float32: { float v; std::memcpy(&v, p, 4); return v; }
        case Scalar::Float64: { double v; std::memcpy(&v, p, 8); return v; }
    }
    throw std::runtime_error("unreachable scalar type");
}

const Property* findProperty(const Element& element, const std::vector<std::string>& names) {
    for (const Property& property : element.properties) {
        if (std::find(names.begin(), names.end(), property.name) != names.end()) return &property;
    }
    return nullptr;
}

} // namespace

Bounds3 TriangleMesh::bounds() const {
    if (vertices.rows() == 0) throw std::runtime_error("mesh has no vertices");
    Bounds3 result;
    result.minimum = vertices.colwise().minCoeff().transpose();
    result.maximum = vertices.colwise().maxCoeff().transpose();
    if (!result.valid()) {
        throw std::runtime_error("mesh bounding box is degenerate; the mesh must span all three axes");
    }
    return result;
}

TriangleMesh readPlyMesh(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open PLY file: " + path.string());

    // ---- header: always ASCII, terminated by "end_header" ----
    std::string line;
    if (!std::getline(stream, line)) throw std::runtime_error("empty PLY file: " + path.string());
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    if (line != "ply") throw std::runtime_error("not a PLY file (missing 'ply' magic): " + path.string());

    Format format = Format::Ascii;
    bool haveFormat = false;
    std::vector<Element> elements;
    long headerLines = 1;
    while (std::getline(stream, line)) {
        ++headerLines;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword == "comment" || keyword == "obj_info") continue;
        if (keyword == "format") {
            std::string name;
            tokens >> name;
            if (name == "ascii") format = Format::Ascii;
            else if (name == "binary_little_endian") format = Format::BinaryLittleEndian;
            else if (name == "binary_big_endian") {
                throw std::runtime_error("PLY big-endian files are not supported: " + path.string());
            } else throw std::runtime_error("unknown PLY format: " + name);
            haveFormat = true;
        } else if (keyword == "element") {
            Element element;
            tokens >> element.name >> element.count;
            if (element.count < 0) throw std::runtime_error("negative PLY element count");
            elements.push_back(std::move(element));
        } else if (keyword == "property") {
            if (elements.empty()) throw std::runtime_error("PLY property before any element");
            std::string type;
            tokens >> type;
            Property property;
            if (type == "list") {
                property.isList = true;
                std::string countType, valueType;
                tokens >> countType >> valueType >> property.name;
                property.countType = parseScalar(countType);
                property.valueType = parseScalar(valueType);
                elements.back().fixedStride = false;
            } else {
                tokens >> property.name;
                property.valueType = parseScalar(type);
            }
            elements.back().properties.push_back(std::move(property));
        } else if (keyword == "end_header") {
            break;
        } else {
            throw std::runtime_error("unrecognized PLY header line: " + line);
        }
    }
    if (!haveFormat) throw std::runtime_error("PLY header has no format line: " + path.string());
    (void)headerLines;

    // Assign fixed-stride offsets for elements without lists.
    for (Element& element : elements) {
        if (!element.fixedStride) continue;
        int offset = 0;
        for (Property& property : element.properties) {
            property.byteOffset = offset;
            offset += scalarSize(property.valueType);
        }
        element.stride = offset;
    }

    const Element* vertexElement = nullptr;
    const Element* faceElement = nullptr;
    for (const Element& element : elements) {
        if (element.name == "vertex") vertexElement = &element;
        else if (element.name == "face") faceElement = &element;
    }
    if (!vertexElement) throw std::runtime_error("PLY file has no 'vertex' element: " + path.string());

    const Property* px = findProperty(*vertexElement, {"x"});
    const Property* py = findProperty(*vertexElement, {"y"});
    const Property* pz = findProperty(*vertexElement, {"z"});
    if (!px || !py || !pz) throw std::runtime_error("PLY vertex element lacks x/y/z");
    const Property* faceIndex = nullptr;
    if (faceElement) {
        faceIndex = findProperty(*faceElement, {"vertex_indices", "vertex_index"});
    }

    TriangleMesh mesh;
    mesh.vertices.resize(vertexElement->count, 3);
    long vertexRow = 0;

    if (format == Format::Ascii) {
        for (const Element& element : elements) {
            const bool isVertex = &element == vertexElement;
            const bool isFace = &element == faceElement;
            for (long index = 0; index < element.count; ++index) {
                double xyz[3] = {0.0, 0.0, 0.0};
                std::vector<long> faceCorners;
                for (const Property& property : element.properties) {
                    if (property.isList) {
                        long listCount = 0;
                        if (!(stream >> listCount)) {
                            throw std::runtime_error("PLY ended mid-record in element '" + element.name + "'");
                        }
                        for (long k = 0; k < listCount; ++k) {
                            double value = 0.0;
                            if (!(stream >> value)) {
                                throw std::runtime_error("PLY ended inside a list property");
                            }
                            if (isFace && &property == faceIndex) faceCorners.push_back(static_cast<long>(value));
                        }
                    } else {
                        double value = 0.0;
                        if (!(stream >> value)) {
                            throw std::runtime_error("PLY ended mid-record in element '" + element.name + "'");
                        }
                        if (isVertex) {
                            if (&property == px) xyz[0] = value;
                            else if (&property == py) xyz[1] = value;
                            else if (&property == pz) xyz[2] = value;
                        }
                    }
                }
                if (isVertex) {
                    mesh.vertices.row(vertexRow++) << xyz[0], xyz[1], xyz[2];
                } else if (isFace && faceIndex) {
                    if (faceCorners.size() != 3) {
                        throw std::runtime_error("PLY face is not a triangle (the field generator needs a triangle mesh)");
                    }
                    mesh.faces.conservativeResize(mesh.faces.rows() + 1, 3);
                    mesh.faces.row(mesh.faces.rows() - 1) << faceCorners[0], faceCorners[1], faceCorners[2];
                }
            }
        }
    } else {
        std::vector<unsigned char> body((std::istreambuf_iterator<char>(stream)),
                                        std::istreambuf_iterator<char>());
        std::size_t cursor = 0;
        auto ensure = [&](std::size_t need, const char* what) {
            if (cursor + need > body.size()) {
                throw std::runtime_error(std::string("PLY binary body is truncated while reading ") + what);
            }
        };
        for (const Element& element : elements) {
            const bool isVertex = &element == vertexElement;
            const bool isFace = &element == faceElement;
            for (long index = 0; index < element.count; ++index) {
                double xyz[3] = {0.0, 0.0, 0.0};
                std::vector<long> faceCorners;
                for (const Property& property : element.properties) {
                    if (property.isList) {
                        ensure(scalarSize(property.countType), "a list count");
                        const long listCount = static_cast<long>(
                            decodeScalar(body.data() + cursor, property.countType));
                        cursor += scalarSize(property.countType);
                        ensure(static_cast<std::size_t>(std::max(0L, listCount)) *
                                   scalarSize(property.valueType), "list values");
                        for (long k = 0; k < listCount; ++k) {
                            const double value = decodeScalar(body.data() + cursor, property.valueType);
                            if (isFace && &property == faceIndex) faceCorners.push_back(static_cast<long>(value));
                            cursor += scalarSize(property.valueType);
                        }
                    } else {
                        ensure(scalarSize(property.valueType), "a property");
                        const double value = decodeScalar(body.data() + cursor, property.valueType);
                        cursor += scalarSize(property.valueType);
                        if (isVertex) {
                            if (&property == px) xyz[0] = value;
                            else if (&property == py) xyz[1] = value;
                            else if (&property == pz) xyz[2] = value;
                        }
                    }
                }
                if (isVertex) {
                    mesh.vertices.row(vertexRow++) << xyz[0], xyz[1], xyz[2];
                } else if (isFace && faceIndex) {
                    if (faceCorners.size() != 3) {
                        throw std::runtime_error("PLY face is not a triangle (the field generator needs a triangle mesh)");
                    }
                    mesh.faces.conservativeResize(mesh.faces.rows() + 1, 3);
                    mesh.faces.row(mesh.faces.rows() - 1) << faceCorners[0], faceCorners[1], faceCorners[2];
                }
            }
        }
    }

    if (mesh.vertices.rows() == 0) throw std::runtime_error("PLY mesh has no vertices");
    if (mesh.faces.rows() == 0) throw std::runtime_error("PLY mesh has no triangular faces");
    if (!mesh.vertices.allFinite()) {
        throw std::runtime_error("PLY mesh contains non-finite vertex coordinates");
    }
    for (Eigen::Index i = 0; i < mesh.faces.rows(); ++i) {
        for (int c = 0; c < 3; ++c) {
            if (mesh.faces(i, c) < 0 || mesh.faces(i, c) >= mesh.vertices.rows()) {
                throw std::runtime_error("PLY face references a vertex index that does not exist");
            }
        }
    }
    return mesh;
}

} // namespace mf
