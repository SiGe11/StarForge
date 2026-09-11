// Terrain.h — terraced heightmap world with cliffs, ramps and slope-based passability.
#pragma once
#include <vector>
#include <cstdint>
#include "../core/Math.h"

namespace sf {

// Interleaved vertex uploaded straight to Metal. Keep in sync with the terrain vertex shader.
struct TerrainVertex {
    float px, py, pz;
    float nx, ny, nz;
    float ao;   // baked ambient occlusion from local relief
    float var;  // low-frequency variation used to break up albedo
};

class Terrain {
public:
    static constexpr int   N     = 128;              // cells per side
    static constexpr float CELL  = 2.0f;             // world units per cell
    static constexpr float SIZE  = N * CELL;         // 256 world units
    static constexpr float HSCALE = 26.0f;           // peak elevation
    static constexpr float WATER = 2.4f;             // sea level
    static constexpr int   VN    = N + 1;            // corner samples per side

    void generate(uint32_t seed, v2 baseA, v2 baseB);

    // --- queries -----------------------------------------------------------
    float heightAt(float x, float z) const;          // bilinear, clamped at edges
    v3    normalAt(float x, float z) const;
    bool  passableCell(int cx, int cz) const {
        if (cx < 0 || cz < 0 || cx >= N || cz >= N) return false;
        return pass_[cz * N + cx] != 0;
    }
    bool  passableWorld(float x, float z) const {
        return passableCell((int)std::floor(x / CELL), (int)std::floor(z / CELL));
    }
    static bool inBoundsWorld(float x, float z) {
        return x >= 0.0f && z >= 0.0f && x < SIZE && z < SIZE;
    }
    float cornerHeight(int x, int z) const {
        x = std::max(0, std::min(VN - 1, x));
        z = std::max(0, std::min(VN - 1, z));
        return h_[z * VN + x];
    }
    // Cheap ray/plane-ish pick: marches the view ray against the height field.
    bool raycast(v3 origin, v3 dir, v3& hit) const;

    // Water surface, emitted only over submerged cells; `ao` carries depth.
    void buildWaterMesh(std::vector<TerrainVertex>& v, std::vector<uint32_t>& i) const;
    // N*N RGBA image of the map for the HUD minimap.
    void buildMinimapRGBA(std::vector<uint8_t>& rgba) const;

    const std::vector<TerrainVertex>& vertices() const { return verts_; }
    const std::vector<uint32_t>&      indices()  const { return idx_; }
    const std::vector<uint8_t>&       passGrid() const { return pass_; }

private:
    void buildMesh();
    void computePassability();
    bool connected(v2 a, v2 b) const;
    void carveCorridor(v2 a, v2 b);
    void flattenDisc(v2 c, float rInner, float rOuter, float targetH);
    float& hRef(int x, int z) { return h_[z * VN + x]; }

    std::vector<float>   h_;      // VN*VN corner heights
    std::vector<uint8_t> pass_;   // N*N passability
    std::vector<TerrainVertex> verts_;
    std::vector<uint32_t>      idx_;
};

} // namespace sf
