#include "Nav.h"
#include <queue>

namespace sf {

void Nav::init(const Terrain* t) {
    terrain_ = t;
    const int NN = Terrain::N * Terrain::N;
    blocked_.assign(NN, 0);
    gScore_.assign(NN, 0.0f);
    came_.assign(NN, -1);
    stamp_.assign(NN, 0);
    run_ = 0;
}

void Nav::addBlocker(v2 c, float r) {
    int x0 = std::max(0, (int)((c.x - r) / Terrain::CELL));
    int x1 = std::min(Terrain::N - 1, (int)((c.x + r) / Terrain::CELL));
    int z0 = std::max(0, (int)((c.y - r) / Terrain::CELL));
    int z1 = std::min(Terrain::N - 1, (int)((c.y + r) / Terrain::CELL));
    for (int z = z0; z <= z1; z++)
        for (int x = x0; x <= x1; x++)
            if (length(cellCentre(x, z) - c) <= r) blocked_[z * Terrain::N + x]++;
}

void Nav::removeBlocker(v2 c, float r) {
    int x0 = std::max(0, (int)((c.x - r) / Terrain::CELL));
    int x1 = std::min(Terrain::N - 1, (int)((c.x + r) / Terrain::CELL));
    int z0 = std::max(0, (int)((c.y - r) / Terrain::CELL));
    int z1 = std::min(Terrain::N - 1, (int)((c.y + r) / Terrain::CELL));
    for (int z = z0; z <= z1; z++)
        for (int x = x0; x <= x1; x++)
            if (length(cellCentre(x, z) - c) <= r && blocked_[z * Terrain::N + x])
                blocked_[z * Terrain::N + x]--;
}

v2 Nav::nearestWalkable(v2 p, int maxRadius) const {
    int cx = (int)(p.x / Terrain::CELL), cz = (int)(p.y / Terrain::CELL);
    if (walkableCell(cx, cz)) return p;
    for (int r = 1; r <= maxRadius; r++) {
        for (int d = -r; d <= r; d++) {
            const int cand[4][2] = {{cx + d, cz - r}, {cx + d, cz + r},
                                    {cx - r, cz + d}, {cx + r, cz + d}};
            for (auto& c : cand)
                if (walkableCell(c[0], c[1])) return cellCentre(c[0], c[1]);
        }
    }
    return p;
}

// Supercover walk: samples every cell the segment touches, not just one per step.
bool Nav::lineOfWalk(v2 a, v2 b) const {
    v2 d = b - a;
    float len = length(d);
    if (len < 1e-4f) return walkableWorld(a);
    int steps = (int)(len / (Terrain::CELL * 0.4f)) + 2;
    for (int i = 0; i <= steps; i++) {
        v2 p = a + d * ((float)i / steps);
        if (!walkableWorld(p)) return false;
    }
    return true;
}

bool Nav::findPath(v2 from, v2 to, std::vector<v2>& out) {
    out.clear();
    if (!terrain_) return false;
    const int N = Terrain::N;

    from = nearestWalkable(from);
    v2 goal = nearestWalkable(to);
    int sx = (int)clampf(from.x / Terrain::CELL, 0, N - 1);
    int sz = (int)clampf(from.y / Terrain::CELL, 0, N - 1);
    int gx = (int)clampf(goal.x / Terrain::CELL, 0, N - 1);
    int gz = (int)clampf(goal.y / Terrain::CELL, 0, N - 1);
    if (!walkableCell(sx, sz)) return false;
    if (sx == gx && sz == gz) { out.push_back(to); return true; }

    // Straight shot: skip the search entirely when nothing is in the way.
    if (lineOfWalk(from, goal)) { out.push_back(goal); return true; }

    run_++;
    auto H = [&](int x, int z) {
        float dx = std::fabs((float)(x - gx)), dz = std::fabs((float)(z - gz));
        // Octile distance: exact for 8-connected movement, so A* stays admissible.
        return (dx + dz) + (1.41421356f - 2.0f) * std::min(dx, dz);
    };

    using QN = std::pair<float, int>;
    std::priority_queue<QN, std::vector<QN>, std::greater<QN>> open;
    int start = sz * N + sx, goalIdx = gz * N + gx;
    gScore_[start] = 0.0f; came_[start] = -1; stamp_[start] = run_;
    open.push({H(sx, sz), start});

    const int dx8[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dz8[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    int best = start; float bestH = H(sx, sz);
    int expanded = 0;
    bool found = false;

    while (!open.empty()) {
        auto [fv, cur] = open.top(); open.pop();
        if (cur == goalIdx) { found = true; break; }
        if (++expanded > 20000) break;      // safety bound on pathological maps
        int cx = cur % N, cz = cur / N;
        float h = H(cx, cz);
        if (h < bestH) { bestH = h; best = cur; }
        for (int i = 0; i < 8; i++) {
            int nx = cx + dx8[i], nz = cz + dz8[i];
            if (!walkableCell(nx, nz)) continue;
            // Refuse to cut a diagonal between two blocked orthogonal neighbours.
            if (i >= 4 && (!walkableCell(cx, nz) || !walkableCell(nx, cz))) continue;
            int ni = nz * N + nx;
            float step = (i < 4) ? 1.0f : 1.41421356f;
            float ng = gScore_[cur] + step;
            if (stamp_[ni] != run_ || ng < gScore_[ni]) {
                stamp_[ni] = run_;
                gScore_[ni] = ng;
                came_[ni] = cur;
                open.push({ng + H(nx, nz), ni});
            }
        }
    }

    int end = found ? goalIdx : best;
    if (end == start) return false;

    std::vector<v2> raw;
    for (int cur = end; cur != -1; cur = came_[cur]) {
        raw.push_back(cellCentre(cur % N, cur / N));
        if (cur == start) break;
    }
    std::reverse(raw.begin(), raw.end());
    if (found) raw.back() = to;

    // String-pull: keep a waypoint only where the direct line would be blocked.
    out.push_back(raw.empty() ? to : raw.front());
    size_t anchor = 0;
    for (size_t i = 1; i < raw.size(); i++) {
        if (!lineOfWalk(raw[anchor], raw[i])) {
            out.push_back(raw[i - 1]);
            anchor = i - 1;
        }
    }
    if (!raw.empty()) out.push_back(raw.back());
    if (out.size() > 1) out.erase(out.begin());   // drop the unit's own cell
    return true;
}

} // namespace sf
