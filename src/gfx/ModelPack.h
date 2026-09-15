// ModelPack.h -- on-disk layout of assets/models.bin, the Blender-authored
// model pack.
//
// The pack is *optional*, exactly like the textures beside it: if the file is
// absent, fails any check below, or omits a mesh, MeshGen.cpp's primitives are
// used for whatever is missing. The game therefore still builds and runs with
// assets/ deleted, and nothing in the build depends on Blender.
//
// Authoring lives in tools/blender/. Do not hand-edit the binary.
//
// Layout, little-endian throughout (both Apple Silicon and x86-64 are LE; the
// loader validates the magic and every count before trusting any offset):
//
//     PackHeader                    32 bytes
//     PackMesh     x meshCount      32 bytes each
//     PackMaterial x materialCount  32 bytes each   -- one palette for all meshes
//     PackVertex   x vertexCount    20 bytes each
//     uint32_t     x indexCount      4 bytes each   -- mesh-local, as MeshRange wants
//
// Twenty bytes per vertex against the 64 the GPU consumes: positions need the
// precision, normals do not (int16 snorm is about 0.003 degrees of error, far
// below what a normal map perturbs anyway), and the eight material floats are
// shared through a palette rather than repeated on every vertex. Expansion to
// MeshVertex happens once at load.
#pragma once
#include <cstdint>

namespace sf {

static constexpr uint32_t kPackMagic   = 0x504D4653u;  // 'SFMP' little-endian
static constexpr uint32_t kPackVersion = 1u;

// Sanity ceilings. These exist so a corrupt header cannot talk the loader into
// a multi-gigabyte allocation before the size check catches it.
static constexpr uint32_t kPackMaxVerts    = 4u * 1000u * 1000u;
static constexpr uint32_t kPackMaxIndices  = 12u * 1000u * 1000u;
static constexpr uint32_t kPackMaxMaterials = 1024u;

struct PackHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t meshCount;
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t materialCount;
    uint32_t reserved[2];
};
static_assert(sizeof(PackHeader) == 32, "PackHeader layout");

struct PackMesh {
    int32_t  meshId;        // a MeshId; anything out of range is skipped
    uint32_t firstVertex;
    uint32_t vertexCount;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t reserved[3];
};
static_assert(sizeof(PackMesh) == 32, "PackMesh layout");

// The MeshVertex tail: albedo, roughness, metallic, team weight, emissive, ao.
struct PackMaterial {
    float cr, cg, cb, rough;
    float metal, team, emis, ao;
};
static_assert(sizeof(PackMaterial) == 32, "PackMaterial layout");

struct PackVertex {
    float    px, py, pz;
    int16_t  nx, ny, nz;    // snorm: value / 32767
    uint16_t material;      // index into the palette
};
static_assert(sizeof(PackVertex) == 20, "PackVertex layout");

} // namespace sf
