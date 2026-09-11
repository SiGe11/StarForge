// Nav.h — grid A* over the terrain's passability, with a dynamic overlay for
// buildings and string-pulled path smoothing.
#pragma once
#include <vector>
#include <cstdint>
#include "../core/Math.h"
#include "Terrain.h"

namespace sf {

class Nav {
public:
    void init(const Terrain* t);

    void addBlocker(v2 centre, float radius);
    void removeBlocker(v2 centre, float radius);

    bool walkableCell(int cx, int cz) const {
        if (cx < 0 || cz < 0 || cx >= Terrain::N || cz >= Terrain::N) return false;
        return terrain_->passableCell(cx, cz) && blocked_[cz * Terrain::N + cx] == 0;
    }
    bool walkableWorld(v2 p) const {
        return walkableCell((int)(p.x / Terrain::CELL), (int)(p.y / Terrain::CELL));
    }
    // Spiral search outward for a usable cell; used when an order lands on a cliff.
    v2 nearestWalkable(v2 p, int maxRadius = 24) const;
    bool lineOfWalk(v2 a, v2 b) const;

    // Returns smoothed waypoints in `out`. Falls back to the reachable cell
    // closest to the goal, so an order onto blocked ground still moves the unit.
    bool findPath(v2 from, v2 to, std::vector<v2>& out);

    static v2 cellCentre(int cx, int cz) {
        return {(cx + 0.5f) * Terrain::CELL, (cz + 0.5f) * Terrain::CELL};
    }

private:
    const Terrain* terrain_ = nullptr;
    std::vector<uint8_t>  blocked_;
    // Scratch reused between queries so pathfinding does not allocate per call.
    std::vector<float>    gScore_;
    std::vector<int32_t>  came_;
    std::vector<uint32_t> stamp_;
    uint32_t              run_ = 0;
};

} // namespace sf
