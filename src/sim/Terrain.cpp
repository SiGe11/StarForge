#include "Terrain.h"
#include "../core/Random.h"
#include <queue>
#include <cmath>

namespace sf {

// A cell is walkable when no corner pair differs by more than this. At CELL=2.0
// this permits roughly a 39-degree incline: ramps pass, terrace cliffs do not.
static constexpr float MAX_STEP = 1.60f;

float Terrain::heightAt(float x, float z) const {
    float gx = clampf(x / CELL, 0.0f, (float)(VN - 1) - 1e-4f);
    float gz = clampf(z / CELL, 0.0f, (float)(VN - 1) - 1e-4f);
    int x0 = (int)gx, z0 = (int)gz;
    float fx = gx - x0, fz = gz - z0;
    float h00 = cornerHeight(x0, z0),     h10 = cornerHeight(x0 + 1, z0);
    float h01 = cornerHeight(x0, z0 + 1), h11 = cornerHeight(x0 + 1, z0 + 1);
    return lerpf(lerpf(h00, h10, fx), lerpf(h01, h11, fx), fz);
}

v3 Terrain::normalAt(float x, float z) const {
    const float e = CELL;
    float hl = heightAt(x - e, z), hr = heightAt(x + e, z);
    float hd = heightAt(x, z - e), hu = heightAt(x, z + e);
    return normalize(v3{hl - hr, 2.0f * e, hd - hu});
}

bool Terrain::raycast(v3 origin, v3 dir, v3& hit) const {
    dir = normalize(dir);
    float t = 0.0f;
    float prevGap = origin.y - heightAt(origin.x, origin.z);
    for (int i = 0; i < 900 && t < 1600.0f; i++) {
        // Longer strides once the ray is far above the surface.
        float step = std::max(0.6f, std::fabs(prevGap) * 0.45f);
        float nt = t + step;
        v3 p = origin + dir * nt;
        if (!inBoundsWorld(p.x, p.z) && p.y < 0.0f) return false;
        float gap = p.y - heightAt(clampf(p.x, 0, SIZE - 0.01f), clampf(p.z, 0, SIZE - 0.01f));
        if (gap <= 0.0f && prevGap > 0.0f) {
            // Bisect the bracketing interval for a clean intersection point.
            float lo = t, hi = nt;
            for (int k = 0; k < 24; k++) {
                float mid = 0.5f * (lo + hi);
                v3 q = origin + dir * mid;
                float g = q.y - heightAt(clampf(q.x, 0, SIZE - 0.01f), clampf(q.z, 0, SIZE - 0.01f));
                if (g > 0.0f) lo = mid; else hi = mid;
            }
            hit = origin + dir * (0.5f * (lo + hi));
            return true;
        }
        prevGap = gap;
        t = nt;
    }
    return false;
}

void Terrain::flattenDisc(v2 c, float rInner, float rOuter, float targetH) {
    int x0 = std::max(0, (int)((c.x - rOuter) / CELL) - 1);
    int x1 = std::min(VN - 1, (int)((c.x + rOuter) / CELL) + 1);
    int z0 = std::max(0, (int)((c.y - rOuter) / CELL) - 1);
    int z1 = std::min(VN - 1, (int)((c.y + rOuter) / CELL) + 1);
    for (int z = z0; z <= z1; z++)
        for (int x = x0; x <= x1; x++) {
            float d = length(v2{x * CELL, z * CELL} - c);
            float w = 1.0f - smoothstepf(rInner, rOuter, d);
            if (w <= 0.0f) continue;
            hRef(x, z) = lerpf(hRef(x, z), targetH, w);
        }
}

void Terrain::computePassability() {
    pass_.assign(N * N, 0);
    for (int z = 0; z < N; z++)
        for (int x = 0; x < N; x++) {
            float a = cornerHeight(x, z),     b = cornerHeight(x + 1, z);
            float c = cornerHeight(x, z + 1), d = cornerHeight(x + 1, z + 1);
            float mn = std::min(std::min(a, b), std::min(c, d));
            float mx = std::max(std::max(a, b), std::max(c, d));
            bool ok = (mx - mn) <= MAX_STEP && mn > WATER + 0.15f;
            pass_[z * N + x] = ok ? 1 : 0;
        }
}

bool Terrain::connected(v2 a, v2 b) const {
    int ax = (int)(a.x / CELL), az = (int)(a.y / CELL);
    int bx = (int)(b.x / CELL), bz = (int)(b.y / CELL);
    if (!passableCell(ax, az) || !passableCell(bx, bz)) return false;
    std::vector<uint8_t> seen(N * N, 0);
    std::vector<int> stack;
    stack.push_back(az * N + ax);
    seen[az * N + ax] = 1;
    const int dx[4] = {1, -1, 0, 0}, dz[4] = {0, 0, 1, -1};
    while (!stack.empty()) {
        int cur = stack.back(); stack.pop_back();
        int cx = cur % N, cz = cur / N;
        if (cx == bx && cz == bz) return true;
        for (int i = 0; i < 4; i++) {
            int nx = cx + dx[i], nz = cz + dz[i];
            if (!passableCell(nx, nz)) continue;
            int ni = nz * N + nx;
            if (seen[ni]) continue;
            seen[ni] = 1;
            stack.push_back(ni);
        }
    }
    return false;
}

// Routes a least-resistance line between the bases (crossing cliffs at high cost)
// then reshapes the height field along it into a walkable ramp.
void Terrain::carveCorridor(v2 a, v2 b) {
    int ax = (int)clampf(a.x / CELL, 0, N - 1), az = (int)clampf(a.y / CELL, 0, N - 1);
    int bx = (int)clampf(b.x / CELL, 0, N - 1), bz = (int)clampf(b.y / CELL, 0, N - 1);

    std::vector<float> dist(N * N, 1e30f);
    std::vector<int>   prev(N * N, -1);
    using QN = std::pair<float, int>;
    std::priority_queue<QN, std::vector<QN>, std::greater<QN>> pq;
    dist[az * N + ax] = 0.0f;
    pq.push({0.0f, az * N + ax});
    const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dz[8] = {0, 0, 1, -1, 1, -1, 1, -1};

    while (!pq.empty()) {
        auto [d, cur] = pq.top(); pq.pop();
        if (d > dist[cur] + 1e-4f) continue;
        if (cur == bz * N + bx) break;
        int cx = cur % N, cz = cur / N;
        for (int i = 0; i < 8; i++) {
            int nx = cx + dx[i], nz = cz + dz[i];
            if (nx < 0 || nz < 0 || nx >= N || nz >= N) continue;
            float step = (i < 4) ? 1.0f : 1.4142f;
            float cost = step;
            if (!pass_[nz * N + nx]) {
                // Water is far more expensive than rock, so corridors prefer land.
                float hh = 0.25f * (cornerHeight(nx, nz) + cornerHeight(nx + 1, nz) +
                                    cornerHeight(nx, nz + 1) + cornerHeight(nx + 1, nz + 1));
                cost += (hh <= WATER + 0.15f) ? 600.0f : 34.0f;
            }
            int ni = nz * N + nx;
            if (dist[cur] + cost < dist[ni]) {
                dist[ni] = dist[cur] + cost;
                prev[ni] = cur;
                pq.push({dist[ni], ni});
            }
        }
    }

    std::vector<int> path;
    for (int cur = bz * N + bx; cur != -1; cur = prev[cur]) {
        path.push_back(cur);
        if (cur == az * N + ax) break;
    }
    if (path.size() < 3) return;
    std::reverse(path.begin(), path.end());

    // Sample the existing profile, then heavily smooth it into a gradual grade.
    const int M = (int)path.size();
    std::vector<float> prof(M);
    for (int i = 0; i < M; i++) {
        int cx = path[i] % N, cz = path[i] / N;
        prof[i] = heightAt((cx + 0.5f) * CELL, (cz + 0.5f) * CELL);
    }
    std::vector<float> tmp(M);
    for (int pass = 0; pass < 40; pass++) {
        for (int i = 0; i < M; i++) {
            float s = prof[i] * 2.0f, w = 2.0f;
            if (i > 0)     { s += prof[i - 1]; w += 1.0f; }
            if (i < M - 1) { s += prof[i + 1]; w += 1.0f; }
            tmp[i] = s / w;
        }
        // Pin the endpoints so the corridor still meets the base plateaus.
        tmp[0] = prof[0]; tmp[M - 1] = prof[M - 1];
        prof.swap(tmp);
    }

    const float R = 3.2f * CELL;   // corridor half-width in world units
    for (int i = 0; i < M; i++) {
        int cx = path[i] % N, cz = path[i] / N;
        v2 c{(cx + 0.5f) * CELL, (cz + 0.5f) * CELL};
        int gx0 = std::max(0, (int)((c.x - R) / CELL)), gx1 = std::min(VN - 1, (int)((c.x + R) / CELL) + 1);
        int gz0 = std::max(0, (int)((c.y - R) / CELL)), gz1 = std::min(VN - 1, (int)((c.y + R) / CELL) + 1);
        for (int z = gz0; z <= gz1; z++)
            for (int x = gx0; x <= gx1; x++) {
                float d = length(v2{x * CELL, z * CELL} - c);
                float w = 1.0f - smoothstepf(R * 0.35f, R, d);
                if (w <= 0.0f) continue;
                hRef(x, z) = lerpf(hRef(x, z), prof[i], w * 0.85f);
            }
    }
}

void Terrain::generate(uint32_t seed, v2 baseA, v2 baseB) {
    h_.assign(VN * VN, 0.0f);
    Rng rng(seed);
    const float ox = rng.range(-800.0f, 800.0f);
    const float oz = rng.range(-800.0f, 800.0f);
    const float L = 3.0f;   // number of terrace levels

    // Pass 1: raw fractal field. Value noise clusters around its midpoint and
    // never reaches 0 or 1, so the actual range is measured and renormalised
    // below -- otherwise the map would have no low ground and no water.
    std::vector<float> raw(VN * VN);
    float rmn = 1e30f, rmx = -1e30f;
    for (int z = 0; z < VN; z++)
        for (int x = 0; x < VN; x++) {
            float nx = x * CELL * 0.0120f + ox, nz = z * CELL * 0.0120f + oz;
            float r = fbm(nx, nz, 5) * 0.74f + ridge(nx * 0.68f + 31.7f, nz * 0.68f - 17.3f, 4) * 0.26f;
            raw[z * VN + x] = r;
            rmn = std::min(rmn, r); rmx = std::max(rmx, r);
        }
    const float inv = 1.0f / std::max(1e-6f, rmx - rmn);

    // Pass 2: normalise, terrace into plateaus, add the rim, scale to world units.
    for (int z = 0; z < VN; z++)
        for (int x = 0; x < VN; x++) {
            float r = saturate((raw[z * VN + x] - rmn) * inv);
            r = std::pow(r, 1.20f);           // bias upward so lakes stay a minority

            // Terrace; the narrow smoothstep band becomes the cliff face.
            float hs = r * L;
            float fl = std::floor(hs);
            float k  = smoothstepf(0.44f, 0.56f, hs - fl);
            float terr = (fl + k) / L;
            float hv = lerpf(terr, r, 0.13f);  // keep plateaus subtly uneven
            hv = 0.02f + hv * 0.98f;           // tier 0 sits below the waterline as lakes

            // Impassable mountain rim so the playfield has hard borders. Ridged
            // noise keeps it from reading as one smooth, obviously synthetic ramp.
            float ex = std::min((float)x, (float)(VN - 1 - x)) / (float)VN;
            float ez = std::min((float)z, (float)(VN - 1 - z)) / (float)VN;
            float rim = 1.0f - smoothstepf(0.025f, 0.15f, std::min(ex, ez));
            float rimH = 1.00f + 0.44f * ridge(x * 0.085f + 55.3f, z * 0.085f - 23.7f, 3);
            hv = lerpf(hv, rimH, rim * rim);

            h_[z * VN + x] = hv * HSCALE;
        }

    // Seat both bases on a flat, above-water plateau snapped to a terrace level.
    auto plateauFor = [&](v2 p) {
        float y = heightAt(p.x, p.y);
        float q = std::round(y / HSCALE * L) / L * HSCALE;
        return std::max(q, WATER + 3.0f);
    };
    flattenDisc(baseA, 17.0f, 33.0f, plateauFor(baseA));
    flattenDisc(baseB, 17.0f, 33.0f, plateauFor(baseB));

    computePassability();
    // Guarantee the two bases can actually reach each other.
    for (int attempt = 0; attempt < 8 && !connected(baseA, baseB); attempt++) {
        carveCorridor(baseA, baseB);
        computePassability();
    }

    buildMesh();
}

void Terrain::buildMesh() {
    verts_.resize(VN * VN);
    for (int z = 0; z < VN; z++)
        for (int x = 0; x < VN; x++) {
            float wx = x * CELL, wz = z * CELL;
            float hc = cornerHeight(x, z);
            v3 n = normalAt(wx, wz);

            // Bake occlusion by comparing against a ring of neighbours: pits and
            // cliff bases darken, exposed ridges stay bright.
            float occ = 0.0f; int cnt = 0;
            const int dx8[8] = {1, -1, 0, 0, 1, 1, -1, -1};
            const int dz8[8] = {0, 0, 1, -1, 1, -1, 1, -1};
            for (int r = 1; r <= 4; r++)
                for (int i = 0; i < 8; i++) {
                    float hh = cornerHeight(x + dx8[i] * r, z + dz8[i] * r);
                    occ += saturate((hh - hc) / (r * CELL * 1.15f));
                    cnt++;
                }
            float ao = 1.0f - saturate(occ / (float)cnt * 1.75f);
            float var = fbm(wx * 0.055f + 91.0f, wz * 0.055f - 44.0f, 4);

            verts_[z * VN + x] = TerrainVertex{wx, hc, wz, n.x, n.y, n.z, ao, var};
        }

    idx_.clear();
    idx_.reserve(N * N * 6);
    for (int z = 0; z < N; z++)
        for (int x = 0; x < N; x++) {
            uint32_t i0 = z * VN + x, i1 = i0 + 1, i2 = i0 + VN, i3 = i2 + 1;
            // Split each quad along its shorter diagonal to keep cliff edges crisp.
            if (std::fabs(h_[i0] - h_[i3]) <= std::fabs(h_[i1] - h_[i2])) {
                idx_.insert(idx_.end(), {i0, i2, i3, i0, i3, i1});
            } else {
                idx_.insert(idx_.end(), {i0, i2, i1, i1, i2, i3});
            }
        }
}

void Terrain::buildWaterMesh(std::vector<TerrainVertex>& V, std::vector<uint32_t>& I) const {
    V.clear(); I.clear();
    for (int z = 0; z < N; z++)
        for (int x = 0; x < N; x++) {
            float c00 = cornerHeight(x, z),     c10 = cornerHeight(x + 1, z);
            float c01 = cornerHeight(x, z + 1), c11 = cornerHeight(x + 1, z + 1);
            float lo = std::min(std::min(c00, c10), std::min(c01, c11));
            // Extend slightly past the shoreline so foam has somewhere to sit.
            if (lo >= WATER + 0.35f) continue;
            const float hs[4] = {c00, c10, c11, c01};
            const int   ox[4] = {0, 1, 1, 0};
            const int   oz[4] = {0, 0, 1, 1};
            uint32_t base = (uint32_t)V.size();
            for (int k = 0; k < 4; k++) {
                float wx = (x + ox[k]) * CELL, wz = (z + oz[k]) * CELL;
                float depth = std::max(0.0f, WATER - hs[k]);
                V.push_back(TerrainVertex{wx, WATER, wz, 0, 1, 0, depth,
                                          fbm(wx * 0.07f, wz * 0.07f, 3)});
            }
            I.insert(I.end(), {base, base + 2, base + 1, base, base + 3, base + 2});
        }
}

void Terrain::buildMinimapRGBA(std::vector<uint8_t>& rgba) const {
    rgba.assign(N * N * 4, 0);
    for (int z = 0; z < N; z++)
        for (int x = 0; x < N; x++) {
            float hgt = heightAt((x + 0.5f) * CELL, (z + 0.5f) * CELL);
            v3 n = normalAt((x + 0.5f) * CELL, (z + 0.5f) * CELL);
            v3 c;
            if (hgt <= WATER) {
                c = lerp(v3{0.05f, 0.18f, 0.26f}, v3{0.10f, 0.32f, 0.40f},
                         saturate((hgt - WATER + 3.0f) / 3.0f));
            } else {
                float t = saturate(hgt / HSCALE);
                c = lerp(v3{0.22f, 0.26f, 0.15f}, v3{0.50f, 0.48f, 0.42f}, t);
                if (!passableCell(x, z)) c = c * 0.55f;   // cliffs read darker
            }
            // Fake a sun rake so relief is legible at minimap scale.
            float shade = 0.72f + 0.45f * saturate(dot(n, normalize(v3{-0.5f, 0.8f, -0.35f})));
            c = c * shade;
            uint8_t* p = &rgba[(z * N + x) * 4];
            p[0] = (uint8_t)(saturate(c.x) * 255.0f);
            p[1] = (uint8_t)(saturate(c.y) * 255.0f);
            p[2] = (uint8_t)(saturate(c.z) * 255.0f);
            p[3] = 255;
        }
}

} // namespace sf
