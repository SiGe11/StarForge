// MeshGen.h — procedural geometry. Every unit, building and prop in the game is
// assembled here from primitives; there are no external art assets.
#pragma once
#include <vector>
#include <cstdint>
#include "RenderTypes.h"

namespace sf {

struct MeshRange {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t  baseVertex = 0;
    float    radius = 1.0f;   // XZ bounding radius, used for culling and rings
    float    height = 1.0f;
};

// Accumulates primitives under a transform stack into one vertex/index soup.
class MeshBuilder {
public:
    std::vector<MeshVertex> verts;
    std::vector<uint32_t>   idx;

    // --- material state applied to subsequently emitted primitives ---------
    v3 col{0.72f, 0.70f, 0.66f};
    float rough = 0.65f, metal = 0.0f, team = 0.0f, emis = 0.0f, ao = 1.0f;
    void mat(v3 c, float r, float m = 0.0f, float t = 0.0f, float e = 0.0f) {
        col = c; rough = r; metal = m; team = t; emis = e;
    }

    // --- transform stack ---------------------------------------------------
    void push(const m4& x) { stack_.push_back(cur_); cur_ = cur_ * x; }
    void pop() { cur_ = stack_.back(); stack_.pop_back(); }
    const m4& xform() const { return cur_; }

    // --- primitives --------------------------------------------------------
    void box(v3 center, v3 half);
    void wedge(v3 center, v3 half, float topShrinkZ, float topShrinkX);
    void cyl(v3 base, float r0, float r1, float h, int seg, bool caps = true);
    void sphere(v3 center, float r, int seg, int rings);
    void ringFlat(v3 center, float rIn, float rOut, int seg);
    void jitterSphere(v3 center, float r, int seg, int rings, uint32_t seed, float amount);

    uint32_t addVert(v3 p, v3 n);
    void tri(uint32_t a, uint32_t b, uint32_t c) { idx.push_back(a); idx.push_back(b); idx.push_back(c); }
    void quad(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { tri(a, b, c); tri(a, c, d); }

private:
    m4 cur_;
    std::vector<m4> stack_;
};

// Emits every mesh in MeshId order into one shared buffer pair.
//
// `modelPackPath`, when non-null, points at a Blender-authored model pack
// (assets/models.bin). Meshes present in the pack replace their procedural
// version; anything missing, or the whole pack if it fails to validate, falls
// back to the primitives below. Passing null is the pure-procedural path, so
// the game still builds and runs with assets/ deleted.
void buildMeshLibrary(std::vector<MeshVertex>& outVerts,
                      std::vector<uint32_t>&   outIdx,
                      MeshRange                outRanges[MESH_COUNT],
                      const char*              modelPackPath = nullptr);

} // namespace sf
