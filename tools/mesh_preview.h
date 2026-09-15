// mesh_preview.h -- software rasteriser and PNG writer for mesh_check.
//
// StarForge renders through Metal, so on anything but a Mac `--shot` is not
// available and a model cannot be looked at. This draws the very same
// MeshVertex data the renderer would upload, with a rough stand-in for the
// object shader, so geometry, normals and materials can be eyeballed on any
// machine. It is a verification aid, not a second renderer: no shadows, no
// fog, no textures, and the tonemap only approximates buildScene's.
//
// No dependencies, in keeping with the rest of the project -- the PNG is
// written with stored (uncompressed) deflate blocks, which needs nothing but
// a CRC32 and an Adler32.
#pragma once
#include "gfx/RenderTypes.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

namespace sf {
namespace preview {

// ---------------------------------------------------------------- PNG
inline uint32_t crc32b(const uint8_t* d, size_t n, uint32_t crc = 0xFFFFFFFFu) {
    static uint32_t tbl[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            tbl[i] = c;
        }
        init = true;
    }
    for (size_t i = 0; i < n; i++) crc = tbl[(crc ^ d[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

inline void be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24)); v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));  v.push_back((uint8_t)x);
}

inline void chunk(std::vector<uint8_t>& out, const char* tag,
                  const std::vector<uint8_t>& data) {
    be32(out, (uint32_t)data.size());
    std::vector<uint8_t> body(tag, tag + 4);
    body.insert(body.end(), data.begin(), data.end());
    out.insert(out.end(), body.begin(), body.end());
    be32(out, crc32b(body.data(), body.size()) ^ 0xFFFFFFFFu);
}

inline bool writePNG(const char* path, int w, int h, const std::vector<uint8_t>& rgb) {
    // Raw scanlines, each prefixed by filter byte 0.
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (w * 3 + 1));
    for (int y = 0; y < h; y++) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + (size_t)y * w * 3,
                   rgb.begin() + (size_t)(y + 1) * w * 3);
    }
    // zlib wrapper around stored deflate blocks: valid, and it means no
    // compressor to depend on.
    std::vector<uint8_t> z;
    z.push_back(0x78); z.push_back(0x01);
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t n = std::min<size_t>(65535, raw.size() - pos);
        bool last = (pos + n) >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back((uint8_t)(n & 0xFF));        z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)(~n & 0xFF));       z.push_back((uint8_t)((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    be32(z, (b << 16) | a);

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    be32(ihdr, (uint32_t)w); be32(ihdr, (uint32_t)h);
    ihdr.push_back(8); ihdr.push_back(2); ihdr.push_back(0);
    ihdr.push_back(0); ihdr.push_back(0);
    chunk(png, "IHDR", ihdr);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", {});

    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    bool ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
    std::fclose(f);
    return ok;
}

// ---------------------------------------------------------------- raster
struct Cam { float yaw, pitch, dist, fov; v3 target; };

struct Framebuffer {
    int w, h;
    std::vector<float> depth;
    std::vector<float> color;   // linear rgb
    // Which pixels ended up showing emissive material. Glowing panels and
    // window bands are the visual signature of these models, and burying one
    // inside the hull it is meant to sit on costs triangles while changing
    // nothing on screen -- a failure with no symptom other than the detail
    // quietly not being there. Tracking coverage makes it checkable.
    std::vector<uint8_t> emisMask;
    Framebuffer(int W, int H) : w(W), h(H), depth((size_t)W * H, 1e30f),
                                color((size_t)W * H * 3, 0.0f),
                                emisMask((size_t)W * H, 0) {}
};

// ---------------------------------------------------------------- shadows
//
// A flat-lit preview is a forgiving preview: without a cast shadow, a shape
// assembled from parts reads as a pile of separately shaded pieces, and a
// silhouette that is actually a box looks acceptable. The game has a shadow
// pass, so judging models without one means judging them in a view that does
// not exist. This is a plain orthographic depth map along the sun direction.
struct ShadowMap {
    int size = 0;
    float extent = 1.0f;
    v3 right, up, dir;      // dir is the direction light travels
    v3 centre;
    std::vector<float> depth;

    bool project(v3 p, float& u, float& v, float& d) const {
        v3 o = p - centre;
        u = (dot(o, right) / extent * 0.5f + 0.5f) * size;
        v = (dot(o, up) / extent * 0.5f + 0.5f) * size;
        d = dot(o, dir);
        return u >= 0 && v >= 0 && u < size && v < size;
    }
};

inline ShadowMap buildShadow(const std::vector<MeshVertex>& V,
                             const std::vector<uint32_t>& I,
                             const MeshRange& R, v3 sunDir, int size = 768) {
    ShadowMap sm;
    sm.size = size;
    sm.dir = normalize(sunDir * -1.0f);
    sm.right = normalize(cross(sm.dir, v3{0, 1, 0}));
    sm.up = cross(sm.right, sm.dir);
    sm.centre = v3{0.0f, R.height * 0.5f, 0.0f};
    sm.extent = std::max({R.radius * 2.4f, R.height * 2.0f, 1.0f});
    sm.depth.assign((size_t)size * size, 1e30f);

    for (uint32_t k = 0; k < R.indexCount; k += 3) {
        float px[3], py[3], pz[3];
        for (int i = 0; i < 3; i++) {
            const MeshVertex& mv = V[I[R.firstIndex + k + i] + R.baseVertex];
            sm.project(v3{mv.px, mv.py, mv.pz}, px[i], py[i], pz[i]);
        }
        float ar = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
        if (std::fabs(ar) < 1e-9f) continue;
        float inv = 1.0f / ar;
        int minx = std::max(0, (int)std::floor(std::min({px[0], px[1], px[2]})));
        int maxx = std::min(size - 1, (int)std::ceil(std::max({px[0], px[1], px[2]})));
        int miny = std::max(0, (int)std::floor(std::min({py[0], py[1], py[2]})));
        int maxy = std::min(size - 1, (int)std::ceil(std::max({py[0], py[1], py[2]})));
        for (int y = miny; y <= maxy; y++)
            for (int x = minx; x <= maxx; x++) {
                float fx = x + 0.5f, fy = y + 0.5f;
                float w0 = ((px[1]-fx)*(py[2]-fy) - (px[2]-fx)*(py[1]-fy)) * inv;
                float w1 = ((px[2]-fx)*(py[0]-fy) - (px[0]-fx)*(py[2]-fy)) * inv;
                float w2 = 1.0f - w0 - w1;
                if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                float d = w0*pz[0] + w1*pz[1] + w2*pz[2];
                size_t di = (size_t)y * size + x;
                if (d < sm.depth[di]) sm.depth[di] = d;
            }
    }
    return sm;
}

inline float sampleShadow(const ShadowMap& sm, v3 p, float ndl,
                          v3 n = v3{0, 0, 0}) {
    if (!sm.size) return 1.0f;
    // Normal offset. A depth bias alone has to grow with the surface slope to
    // stop self-shadowing, and by the time it is large enough for a face
    // nearly edge-on to the light it is large enough to detach contact
    // shadows elsewhere. Moving the sample point off the surface along its
    // own normal by about a texel scales correctly by construction, and is
    // what stops large flat plates striping with acne.
    float texel = sm.extent * 2.0f / (float)sm.size;
    float slope = std::sqrt(std::max(0.0f, 1.0f - ndl * ndl)) / std::max(ndl, 0.15f);
    v3 sp = p + n * (texel * (1.2f + 1.8f * std::min(slope, 3.0f)));
    float u, v, d;
    if (!sm.project(sp, u, v, d)) return 1.0f;
    float bias = texel * 0.75f;
    float lit = 0.0f;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            int x = (int)u + i, y = (int)v + j;
            if (x < 0 || y < 0 || x >= sm.size || y >= sm.size) { lit += 1.0f; continue; }
            lit += (d - bias <= sm.depth[(size_t)y * sm.size + x]) ? 1.0f : 0.0f;
        }
    return lit / 9.0f;
}

// Draws a ground plane under the model with a soft contact darkening. Without
// it every model floats in sky and its scale and footprint are unreadable.
inline void drawGround(Framebuffer& fb, const Cam& cam, float ox, float oy,
                       float scl, float radius, const ShadowMap* sm = nullptr) {
    float cy = std::cos(cam.yaw), sy = std::sin(cam.yaw);
    float cp = std::cos(cam.pitch), sp = std::sin(cam.pitch);
    v3 fwd{ -sy * cp, -sp, -cy * cp };
    v3 eye = cam.target - fwd * cam.dist;
    v3 right = normalize(cross(fwd, v3{0, 1, 0}));
    v3 up = cross(right, fwd);
    float f = 1.0f / std::tan(cam.fov * 0.5f);

    int x0 = (int)(ox - scl * 1.12f), x1 = (int)(ox + scl * 1.12f);
    int y0 = (int)(oy - scl * 1.05f), y1 = (int)(oy + scl * 1.05f);
    x0 = std::max(0, x0); y0 = std::max(0, y0);
    x1 = std::min(fb.w - 1, x1); y1 = std::min(fb.h - 1, y1);

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            // Unproject the pixel onto the y=0 plane.
            float nx = (x + 0.5f - ox) / (scl * f);
            float ny = -(y + 0.5f - oy) / (scl * f);
            v3 dir = fwd + right * nx + up * ny;
            if (dir.y >= -1e-4f) continue;
            float t = -eye.y / dir.y;
            if (t <= 0.01f) continue;
            v3 hit = eye + dir * t;
            float d = std::sqrt(hit.x * hit.x + hit.z * hit.z);
            if (d > radius * 2.3f) continue;
            size_t di = (size_t)y * fb.w + x;
            if (t >= fb.depth[di]) continue;
            fb.depth[di] = t;
            float fade = saturate(1.0f - d / (radius * 2.3f));
            float contact = saturate(1.0f - d / (radius * 1.15f));
            // shadeAndDraw writes tonemapped, gamma-encoded values into this
            // buffer, so the ground has to be encoded the same way or a
            // perfectly ordinary linear 0.3 renders as near-black.
            float g = 0.34f * (0.35f + 0.65f * fade) * (1.0f - 0.25f * contact * contact);
            // The cast shadow on the ground is most of what makes a
            // silhouette legible, so it is worth more here than anywhere.
            if (sm) g *= 0.30f + 0.70f * sampleShadow(*sm, hit, 0.7f, v3{0, 1, 0});
            auto enc = [](float v) { return std::pow(saturate(v), 1.0f / 2.2f); };
            fb.color[di * 3 + 0] = enc(g * 1.00f);
            fb.color[di * 3 + 1] = enc(g * 0.97f);
            fb.color[di * 3 + 2] = enc(g * 0.88f);
        }
    }
}

inline void clearSky(Framebuffer& fb) {
    for (int y = 0; y < fb.h; y++) {
        float t = (float)y / (float)fb.h;
        // Rough match for skyColor's horizon gradient, purely so the
        // silhouette reads against something.
        float r = lerpf(0.26f, 0.52f, t), g = lerpf(0.36f, 0.58f, t),
              b = lerpf(0.55f, 0.62f, t);
        for (int x = 0; x < fb.w; x++) {
            size_t i = ((size_t)y * fb.w + x) * 3;
            fb.color[i] = r; fb.color[i + 1] = g; fb.color[i + 2] = b;
        }
    }
}

// The game's sun sits roughly perpendicular to the default view axis, which
// per CLAUDE.md is what makes shadows visible at all; the same choice here
// keeps one side of a model lit and the other in ambient.
inline void shadeAndDraw(Framebuffer& fb,
                         const std::vector<MeshVertex>& V,
                         const std::vector<uint32_t>& I,
                         const MeshRange& R, const Cam& cam,
                         v3 teamColor, float ox, float oy, float scl,
                         const ShadowMap* sm = nullptr) {
    const v3 sunDir = normalize(v3{0.62f, 0.66f, 0.42f});
    // Calibrated the way CLAUDE.md describes buildScene's set: a ~0.18 albedo
    // must land near mid-grey once ACES and gamma have been applied. Guessing
    // these individually is exactly the mistake that note warns about --
    // at sunIntensity 3.1 every armour surface clipped to white and the
    // bevels the models were rebuilt for became invisible.
    const float sunI = 1.25f, ambient = 0.22f, exposure = 1.0f;

    float cy = std::cos(cam.yaw), sy = std::sin(cam.yaw);
    float cp = std::cos(cam.pitch), sp = std::sin(cam.pitch);
    v3 fwd{ -sy * cp, -sp, -cy * cp };
    v3 eye = cam.target - fwd * cam.dist;
    v3 right = normalize(cross(fwd, v3{0, 1, 0}));
    v3 up = cross(right, fwd);

    auto project = [&](v3 p, float& sx, float& sy_, float& z) {
        v3 d = p - eye;
        float vx = dot(d, right), vy = dot(d, up), vz = dot(d, fwd);
        z = vz;
        if (vz < 0.01f) return false;
        float f = 1.0f / std::tan(cam.fov * 0.5f);
        sx = ox + (vx / vz) * f * scl;
        sy_ = oy - (vy / vz) * f * scl;
        return true;
    };

    for (uint32_t k = 0; k < R.indexCount; k += 3) {
        const MeshVertex* tv[3] = {
            &V[I[R.firstIndex + k + 0] + R.baseVertex],
            &V[I[R.firstIndex + k + 1] + R.baseVertex],
            &V[I[R.firstIndex + k + 2] + R.baseVertex],
        };
        float px[3], py[3], pz[3];
        bool ok = true;
        for (int i = 0; i < 3; i++)
            ok &= project(v3{tv[i]->px, tv[i]->py, tv[i]->pz}, px[i], py[i], pz[i]);
        if (!ok) continue;

        // Backface cull in screen space, matching cullMode: back with CCW
        // front faces -- so a model that renders here is one that renders in
        // the game, and an inside-out mesh shows up as holes here too.
        float ar = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
        if (ar >= 0.0f) continue;

        int minx = std::max(0, (int)std::floor(std::min({px[0], px[1], px[2]})));
        int maxx = std::min(fb.w - 1, (int)std::ceil(std::max({px[0], px[1], px[2]})));
        int miny = std::max(0, (int)std::floor(std::min({py[0], py[1], py[2]})));
        int maxy = std::min(fb.h - 1, (int)std::ceil(std::max({py[0], py[1], py[2]})));
        if (minx > maxx || miny > maxy) continue;
        float inv = 1.0f / ar;

        for (int y = miny; y <= maxy; y++) {
            for (int x = minx; x <= maxx; x++) {
                float fx = x + 0.5f, fy = y + 0.5f;
                float w0 = ((px[1] - fx) * (py[2] - fy) - (px[2] - fx) * (py[1] - fy)) * inv;
                float w1 = ((px[2] - fx) * (py[0] - fy) - (px[0] - fx) * (py[2] - fy)) * inv;
                float w2 = 1.0f - w0 - w1;
                if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                float z = w0 * pz[0] + w1 * pz[1] + w2 * pz[2];
                size_t di = (size_t)y * fb.w + x;
                if (z >= fb.depth[di]) continue;
                fb.depth[di] = z;

                v3 n = normalize(v3{
                    w0 * tv[0]->nx + w1 * tv[1]->nx + w2 * tv[2]->nx,
                    w0 * tv[0]->ny + w1 * tv[1]->ny + w2 * tv[2]->ny,
                    w0 * tv[0]->nz + w1 * tv[1]->nz + w2 * tv[2]->nz});
                v3 alb{w0 * tv[0]->cr + w1 * tv[1]->cr + w2 * tv[2]->cr,
                       w0 * tv[0]->cg + w1 * tv[1]->cg + w2 * tv[2]->cg,
                       w0 * tv[0]->cb + w1 * tv[1]->cb + w2 * tv[2]->cb};
                float rough = w0 * tv[0]->rough + w1 * tv[1]->rough + w2 * tv[2]->rough;
                float team  = w0 * tv[0]->team  + w1 * tv[1]->team  + w2 * tv[2]->team;
                float emis  = w0 * tv[0]->emis  + w1 * tv[1]->emis  + w2 * tv[2]->emis;
                float ao    = w0 * tv[0]->ao    + w1 * tv[1]->ao    + w2 * tv[2]->ao;

                alb = alb * (1.0f - team) + teamColor * team;

                float ndl = std::max(0.0f, dot(n, sunDir));
                v3 wp{w0*tv[0]->px + w1*tv[1]->px + w2*tv[2]->px,
                      w0*tv[0]->py + w1*tv[1]->py + w2*tv[2]->py,
                      w0*tv[0]->pz + w1*tv[1]->pz + w2*tv[2]->pz};
                if (sm) ndl *= sampleShadow(*sm, wp, ndl, n);
                v3 vdir = normalize(eye - v3{0, 0, 0});
                v3 hv = normalize(sunDir + vdir);
                float spec = std::pow(std::max(0.0f, dot(n, hv)),
                                      std::max(2.0f, 2.0f / (rough * rough + 1e-3f)));
                float sky = 0.5f + 0.5f * n.y;

                // Weak fill from the opposite side. Without it every surface
                // facing away from the sun collapses to one ambient value and
                // the shaded half of a model loses all its form.
                const v3 fillDir = normalize(v3{-0.55f, 0.35f, -0.75f});
                float fill = std::max(0.0f, dot(n, fillDir)) * 0.30f;
                v3 c = alb * (ndl * sunI + fill + ambient * sky * ao)
                     + v3{1.0f, 0.96f, 0.9f} * (spec * (1.0f - rough) * sunI * 0.5f)
                     + alb * emis;
                c = c * exposure;
                // ACES-ish curve, then gamma, so the preview sits in roughly
                // the same range as the shipped tonemap.
                auto tm = [](float v) {
                    v = (v * (2.51f * v + 0.03f)) / (v * (2.43f * v + 0.59f) + 0.14f);
                    return std::pow(saturate(v), 1.0f / 2.2f);
                };
                fb.color[di * 3 + 0] = tm(c.x);
                fb.color[di * 3 + 1] = tm(c.y);
                fb.color[di * 3 + 2] = tm(c.z);
                fb.emisMask[di] = (emis > 0.5f) ? 1 : 0;
            }
        }
    }
}

// How many emissive triangles a mesh declares, and how many pixels of it
// actually survive to the screen across a full orbit. Zero pixels against a
// non-zero triangle count means the detail is inside something.
struct EmisReport { int tris = 0; long pixels = 0; };

inline EmisReport checkEmissive(const std::vector<MeshVertex>& V,
                                const std::vector<uint32_t>& I,
                                const MeshRange& R) {
    EmisReport rep;
    for (uint32_t k = 0; k < R.indexCount; k += 3) {
        const MeshVertex& A = V[I[R.firstIndex + k] + R.baseVertex];
        if (A.emis > 0.5f) rep.tris++;
    }
    if (!rep.tris) return rep;

    // Four yaws and two pitches: one unlucky angle should not condemn a panel
    // that is simply facing away.
    const int S = 128;
    for (int a = 0; a < 4; a++) {
        for (int p = 0; p < 2; p++) {
            Framebuffer fb(S, S);
            Cam cam;
            cam.yaw = a * 1.5707963f;
            cam.pitch = (p == 0) ? 0.25f : 0.70f;
            float extent = std::max({R.radius, R.height * 0.75f, 0.5f});
            cam.dist = extent * 2.7f;
            cam.fov = 0.9f;
            cam.target = v3{0.0f, R.height * 0.42f, 0.0f};
            shadeAndDraw(fb, V, I, R, cam, v3{0.3f, 0.55f, 0.95f},
                         S * 0.5f, S * 0.5f, S * 0.5f);
            for (uint8_t m : fb.emisMask) rep.pixels += m;
        }
    }
    return rep;
}

inline std::vector<uint8_t> resolve(const Framebuffer& fb, int ss) {
    int w = fb.w / ss, h = fb.h / ss;
    std::vector<uint8_t> out((size_t)w * h * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) {
                float a = 0;
                for (int j = 0; j < ss; j++)
                    for (int i = 0; i < ss; i++)
                        a += fb.color[(((size_t)(y * ss + j) * fb.w) + x * ss + i) * 3 + c];
                out[((size_t)y * w + x) * 3 + c] =
                    (uint8_t)std::lround(saturate(a / (ss * ss)) * 255.0f);
            }
    return out;
}

} // namespace preview
} // namespace sf
