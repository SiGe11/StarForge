// Random.h — deterministic PRNG and value-noise helpers shared by sim and worldgen.
#pragma once
#include <cstdint>
#include <cmath>
#include "Math.h"

namespace sf {

struct Rng {
    uint64_t s = 0x853c49e6748fea9bULL;
    explicit Rng(uint64_t seed = 12345) { s = seed * 6364136223846793005ULL + 1442695040888963407ULL; next(); }
    uint32_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return (uint32_t)((s * 2685821657736338717ULL) >> 32);
    }
    float f01() { return (float)(next() & 0xFFFFFF) / (float)0x1000000; }
    float range(float a, float b) { return a + (b - a) * f01(); }
    int  irange(int a, int b) { return b <= a ? a : a + (int)(next() % (uint32_t)(b - a)); }
    v2   inUnitCircle() {
        for (int i = 0; i < 8; i++) {
            v2 p{range(-1, 1), range(-1, 1)};
            if (length2(p) <= 1.0f) return p;
        }
        return {0, 0};
    }
};

// --- deterministic hash-based value noise (no tables, no allocation) --------
inline float hash2(int x, int y) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return (float)((h ^ (h >> 16)) & 0xFFFFFF) / (float)0x1000000;
}
inline float valueNoise(float x, float y) {
    int xi = (int)std::floor(x), yi = (int)std::floor(y);
    float fx = x - xi, fy = y - yi;
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    float a = hash2(xi, yi),     b = hash2(xi + 1, yi);
    float c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
    return lerpf(lerpf(a, b, ux), lerpf(c, d, ux), uy);
}
inline float fbm(float x, float y, int octaves = 5, float lac = 2.03f, float gain = 0.5f) {
    float sum = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < octaves; i++) {
        sum  += amp * valueNoise(x, y);
        norm += amp;
        x *= lac; y *= lac; amp *= gain;
    }
    return sum / norm;
}
// Ridged variant gives mountain spines that read well under a low sun.
inline float ridge(float x, float y, int octaves = 4) {
    float sum = 0, amp = 0.5f, norm = 0;
    for (int i = 0; i < octaves; i++) {
        float n = 1.0f - std::fabs(valueNoise(x, y) * 2.0f - 1.0f);
        sum += amp * n * n;
        norm += amp;
        x *= 2.07f; y *= 2.07f; amp *= 0.5f;
    }
    return sum / norm;
}

} // namespace sf
