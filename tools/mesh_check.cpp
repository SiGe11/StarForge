// mesh_check.cpp — offline validation of the mesh library.
//
// Builds the same vertex/index soup the renderer uploads and audits it without
// a GPU, so a bad mesh is caught on any machine rather than showing up as an
// invisible model on a Mac. It answers the check CLAUDE.md asks for after any
// MeshBuilder edit -- for every triangle the geometric normal cross(b-a, c-a)
// must agree with the average of the three stored vertex normals -- plus the
// budget numbers that decide whether a model is affordable at --stress.
//
//   clang++ -std=c++20 -O2 -I src tools/mesh_check.cpp src/gfx/MeshGen.cpp -o /tmp/mc
//   /tmp/mc [path/to/models.bin] [--png sheet.png]
//
// Exit status is non-zero if any mesh fails, so it works as a pre-commit gate.
//
// `--png` renders a contact sheet of every mesh through the small software
// rasteriser in mesh_preview.h. The game itself needs Metal, so on a machine
// without one this is the only way to look at a model at all; on a Mac it is
// still the quickest way to see all twelve at once without starting a match.

#include "gfx/MeshGen.h"
#include "mesh_preview.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <cstdlib>

using namespace sf;

static const char* kNames[MESH_COUNT] = {
    "WORKER", "TROOPER", "MAULER_HULL", "MAULER_TURRET",
    "FOUNDRY", "GARRISON", "WORKSHOP", "BUNKHOUSE",
    "ORE", "BOULDER", "PROJECTILE", "SEL_RING"
};

// Meshes that are deliberately flat sheets: a ring drawn with cullMode none has
// no outside, so the winding rule does not apply to it.
static bool isFlatSheet(int m) { return m == MESH_SEL_RING; }

struct MeshAudit {
    int tris = 0, verts = 0;
    int degenerate = 0;      // zero-area triangles
    int backfacing = 0;      // geometric normal opposes the shading normal
    int borderline = 0;      // agreement below 0.2 -- not wrong, but very thin
    float worstDot = 1.0f;
    float minY = 1e30f, maxY = -1e30f;
    float maxR = 0.0f;
    double area = 0.0;
};

static MeshAudit auditMesh(const std::vector<MeshVertex>& V,
                           const std::vector<uint32_t>& I,
                           const MeshRange& R, bool flat) {
    MeshAudit a;
    a.tris = (int)(R.indexCount / 3);

    std::map<uint32_t, int> seen;
    for (uint32_t k = 0; k < R.indexCount; k += 3) {
        uint32_t i0 = I[R.firstIndex + k + 0] + (uint32_t)R.baseVertex;
        uint32_t i1 = I[R.firstIndex + k + 1] + (uint32_t)R.baseVertex;
        uint32_t i2 = I[R.firstIndex + k + 2] + (uint32_t)R.baseVertex;
        if (i0 >= V.size() || i1 >= V.size() || i2 >= V.size()) {
            fprintf(stderr, "  index out of range\n");
            a.degenerate++;
            continue;
        }
        seen[i0] = seen[i1] = seen[i2] = 1;

        const MeshVertex& A = V[i0];
        const MeshVertex& B = V[i1];
        const MeshVertex& C = V[i2];
        for (const MeshVertex* p : {&A, &B, &C}) {
            a.minY = std::min(a.minY, p->py);
            a.maxY = std::max(a.maxY, p->py);
            a.maxR = std::max(a.maxR, std::sqrt(p->px * p->px + p->pz * p->pz));
        }

        float ex = B.px - A.px, ey = B.py - A.py, ez = B.pz - A.pz;
        float fx = C.px - A.px, fy = C.py - A.py, fz = C.pz - A.pz;
        float hx = C.px - B.px, hy = C.py - B.py, hz = C.pz - B.pz;
        float gx = ey * fz - ez * fy;
        float gy = ez * fx - ex * fz;
        float gz = ex * fy - ey * fx;
        float glen = std::sqrt(gx * gx + gy * gy + gz * gz);
        a.area += 0.5 * glen;
        // Degeneracy has to be judged against the triangle's own size, not an
        // absolute epsilon. A sliver at a sphere pole has two vertices that
        // differ only by sin/cos rounding, so its cross product is ~1e-8 of
        // noise -- large enough to clear a fixed threshold, and pointing in an
        // arbitrary direction, which reads as "backfacing" if it is let
        // through. glen is twice the area and so scales as length squared;
        // comparing it to the longest edge squared is the scale-free test.
        float e2max = std::max({ex*ex + ey*ey + ez*ez,
                                fx*fx + fy*fy + fz*fz,
                                hx*hx + hy*hy + hz*hz});
        if (glen < 1e-6f * e2max || e2max < 1e-20f) { a.degenerate++; continue; }
        gx /= glen; gy /= glen; gz /= glen;

        float sx = A.nx + B.nx + C.nx;
        float sy = A.ny + B.ny + C.ny;
        float sz = A.nz + B.nz + C.nz;
        float slen = std::sqrt(sx * sx + sy * sy + sz * sz);
        if (slen < 1e-6f) { a.degenerate++; continue; }
        sx /= slen; sy /= slen; sz /= slen;

        float d = gx * sx + gy * sy + gz * sz;
        if (flat) d = std::fabs(d);        // a sheet may be seen from either side
        a.worstDot = std::min(a.worstDot, d);
        if (d < 0.0f) a.backfacing++;
        else if (d < 0.2f) a.borderline++;

    }
    a.verts = (int)seen.size();
    return a;
}

int main(int argc, char** argv) {
    const char* pack = nullptr;
    const char* png = nullptr;
    int only = -1;                 // --mesh N renders that one mesh full frame
    float yaw = 0.60f, pitch = 0.42f;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--png") && i + 1 < argc) png = argv[++i];
        else if (!std::strcmp(argv[i], "--mesh") && i + 1 < argc) only = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--yaw") && i + 1 < argc) yaw = (float)atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--pitch") && i + 1 < argc) pitch = (float)atof(argv[++i]);
        else if (argv[i][0] != '-') pack = argv[i];
    }

    std::vector<MeshVertex> V;
    std::vector<uint32_t> I;
    MeshRange R[MESH_COUNT];
    buildMeshLibrary(V, I, R, pack);

    printf("source: %s\n\n", pack ? pack : "procedural (MeshGen.cpp)");
    printf("%-15s %7s %7s %8s %8s %7s %7s  %s\n",
           "mesh", "verts", "tris", "radius", "height", "minY", "worst", "status");
    printf("%s\n", std::string(86, '-').c_str());

    int totalTris = 0, failures = 0;
    for (int m = 0; m < MESH_COUNT; m++) {
        MeshAudit a = auditMesh(V, I, R[m], isFlatSheet(m));
        totalTris += a.tris;

        char status[128];
        if (a.backfacing) {
            snprintf(status, sizeof status, "FAIL %d backfacing, %d degenerate",
                     a.backfacing, a.degenerate);
            failures++;
        } else if (a.degenerate) {
            // Zero-area triangles rasterise to nothing, so they are waste
            // rather than corruption -- worth reporting, not worth failing on.
            snprintf(status, sizeof status, "ok (%d degenerate)", a.degenerate);
        } else if (a.borderline) {
            snprintf(status, sizeof status, "ok (%d thin)", a.borderline);
        } else {
            snprintf(status, sizeof status, "ok");
        }

        printf("%-15s %7d %7d %8.2f %8.2f %7.2f %7.2f  %s\n",
               kNames[m], a.verts, a.tris, R[m].radius, R[m].height,
               a.minY, a.worstDot, status);

        sf::preview::EmisReport er = sf::preview::checkEmissive(V, I, R[m]);
        if (er.tris && er.pixels < 8)
            printf("%-15s   note: %d emissive triangles but %ld visible pixels over a "
                   "full orbit -- glowing detail is buried inside the hull\n",
                   "", er.tris, er.pixels);

        // The renderer places models with their base on the ground and
        // emitMesh derives height from max py, so a model hanging below the
        // origin sinks into the terrain.
        if (a.minY < -0.05f)
            printf("%-15s   note: geometry %.2f below origin, will clip into terrain\n",
                   "", a.minY);
    }

    printf("%s\n", std::string(86, '-').c_str());
    printf("%-15s %7zu %7d  buffers: %.1f KB verts + %.1f KB idx\n", "TOTAL",
           V.size(), totalTris,
           V.size() * sizeof(MeshVertex) / 1024.0,
           I.size() * sizeof(uint32_t) / 1024.0);

    // A rough load figure: what a 200-unit-per-side stress test costs in
    // triangles if every unit is the most expensive one.
    int unitMax = 0;
    for (int m = MESH_WORKER; m <= MESH_MAULER_TURRET; m++)
        unitMax = std::max(unitMax, (int)(R[m].indexCount / 3));
    printf("%-15s %d tris/unit worst case -> %d tris at --stress 200 per side\n",
           "load", unitMax, unitMax * 400);

    if (png) {
        using namespace sf::preview;
        // 4x supersampled, then boxed down: no MSAA here, and bevels are
        // exactly the thin features that alias into invisibility.
        const bool single = (only >= 0 && only < MESH_COUNT);
        const int ss = single ? 3 : 4;
        const int cols = single ? 1 : 4;
        const int rows = single ? 1 : (MESH_COUNT + cols - 1) / cols;
        const int cw = single ? 900 : 320, chh = single ? 760 : 300;
        Framebuffer fb(cw * cols * ss, chh * rows * ss);
        clearSky(fb);
        for (int m = 0; m < MESH_COUNT; m++) {
            if (single && m != only) continue;
            int cx = single ? 0 : m % cols, cy = single ? 0 : m / cols;
            // Frame each model on its own size so a trooper and a foundry are
            // both legible in the same sheet.
            float extent = std::max({R[m].radius, R[m].height * 0.75f, 0.5f});
            Cam cam;
            cam.yaw = yaw;
            cam.pitch = pitch;
            cam.dist = extent * 2.7f;
            cam.fov = 0.9f;
            cam.target = v3{0.0f, R[m].height * 0.42f, 0.0f};
            float ox = (cx + 0.5f) * cw * ss, oy = (cy + 0.5f) * chh * ss;
            float scl = (float)chh * ss * 0.5f;
            drawGround(fb, cam, ox, oy, scl, std::max(R[m].radius, 0.6f));
            shadeAndDraw(fb, V, I, R[m], cam, v3{0.30f, 0.55f, 0.95f}, ox, oy, scl);
        }
        std::vector<uint8_t> rgb = resolve(fb, ss);
        if (writePNG(png, cw * cols, chh * rows, rgb))
            printf("\nwrote %s (%dx%d)\n", png, cw * cols, chh * rows);
        else
            printf("\ncould not write %s\n", png);
    }

    if (failures) {
        printf("\n%d mesh(es) FAILED validation\n", failures);
        return 1;
    }
    printf("\nall meshes passed\n");
    return 0;
}
