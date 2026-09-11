#include "MeshGen.h"
#include "../core/Random.h"

namespace sf {

// ---------------------------------------------------------------- primitives
uint32_t MeshBuilder::addVert(v3 p, v3 n) {
    v3 wp = transformPoint(cur_, p);
    v3 wn = normalize(transformDir(cur_, n));
    MeshVertex mv{};
    mv.px = wp.x; mv.py = wp.y; mv.pz = wp.z;
    mv.nx = wn.x; mv.ny = wn.y; mv.nz = wn.z;
    mv.cr = col.x; mv.cg = col.y; mv.cb = col.z; mv.rough = rough;
    mv.metal = metal; mv.team = team; mv.emis = emis; mv.ao = ao;
    verts.push_back(mv);
    return (uint32_t)verts.size() - 1;
}

void MeshBuilder::box(v3 c, v3 h) {
    const v3 n[6] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
    // Corner offsets per face, wound counter-clockwise when seen from outside.
    const float s[6][4][3] = {
        {{ 1,-1, 1},{ 1,-1,-1},{ 1, 1,-1},{ 1, 1, 1}},
        {{-1,-1,-1},{-1,-1, 1},{-1, 1, 1},{-1, 1,-1}},
        {{-1, 1, 1},{ 1, 1, 1},{ 1, 1,-1},{-1, 1,-1}},
        {{-1,-1,-1},{ 1,-1,-1},{ 1,-1, 1},{-1,-1, 1}},
        {{-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}},
        {{ 1,-1,-1},{-1,-1,-1},{-1, 1,-1},{ 1, 1,-1}},
    };
    for (int f = 0; f < 6; f++) {
        uint32_t v0 = addVert({c.x + s[f][0][0]*h.x, c.y + s[f][0][1]*h.y, c.z + s[f][0][2]*h.z}, n[f]);
        uint32_t v1 = addVert({c.x + s[f][1][0]*h.x, c.y + s[f][1][1]*h.y, c.z + s[f][1][2]*h.z}, n[f]);
        uint32_t v2 = addVert({c.x + s[f][2][0]*h.x, c.y + s[f][2][1]*h.y, c.z + s[f][2][2]*h.z}, n[f]);
        uint32_t v3_ = addVert({c.x + s[f][3][0]*h.x, c.y + s[f][3][1]*h.y, c.z + s[f][3][2]*h.z}, n[f]);
        quad(v0, v1, v2, v3_);
    }
}

// Box whose top face is inset, giving armour plating a bevelled, sloped read.
void MeshBuilder::wedge(v3 c, v3 h, float topShrinkZ, float topShrinkX) {
    float bx = h.x, bz = h.z, tx = h.x * (1.0f - topShrinkX), tz = h.z * (1.0f - topShrinkZ);
    v3 b[4] = {{c.x-bx, c.y-h.y, c.z+bz}, {c.x+bx, c.y-h.y, c.z+bz}, {c.x+bx, c.y-h.y, c.z-bz}, {c.x-bx, c.y-h.y, c.z-bz}};
    v3 t[4] = {{c.x-tx, c.y+h.y, c.z+tz}, {c.x+tx, c.y+h.y, c.z+tz}, {c.x+tx, c.y+h.y, c.z-tz}, {c.x-tx, c.y+h.y, c.z-tz}};
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        v3 nrm = normalize(cross(b[j] - b[i], t[j] - b[i]));
        quad(addVert(b[i], nrm), addVert(b[j], nrm), addVert(t[j], nrm), addVert(t[i], nrm));
    }
    quad(addVert(t[0], {0,1,0}), addVert(t[1], {0,1,0}), addVert(t[2], {0,1,0}), addVert(t[3], {0,1,0}));
    quad(addVert(b[3], {0,-1,0}), addVert(b[2], {0,-1,0}), addVert(b[1], {0,-1,0}), addVert(b[0], {0,-1,0}));
}

void MeshBuilder::cyl(v3 base, float r0, float r1, float h, int seg, bool caps) {
    if (seg < 3) seg = 3;
    float slope = (r0 - r1) / std::max(1e-4f, h);
    for (int i = 0; i < seg; i++) {
        float a0 = TAU * i / seg, a1 = TAU * (i + 1) / seg;
        v3 d0{std::cos(a0), 0, std::sin(a0)}, d1{std::cos(a1), 0, std::sin(a1)};
        v3 n0 = normalize(v3{d0.x, slope, d0.z}), n1 = normalize(v3{d1.x, slope, d1.z});
        v3 p00 = base + d0 * r0, p10 = base + d1 * r0;
        v3 p01 = base + v3{0, h, 0} + d0 * r1, p11 = base + v3{0, h, 0} + d1 * r1;
        quad(addVert(p00, n0), addVert(p01, n0), addVert(p11, n1), addVert(p10, n1));
    }
    if (!caps) return;
    if (r1 > 1e-4f) {
        uint32_t c = addVert(base + v3{0, h, 0}, {0, 1, 0});
        for (int i = 0; i < seg; i++) {
            float a0 = TAU * i / seg, a1 = TAU * (i + 1) / seg;
            tri(c, addVert(base + v3{std::cos(a1)*r1, h, std::sin(a1)*r1}, {0,1,0}),
                   addVert(base + v3{std::cos(a0)*r1, h, std::sin(a0)*r1}, {0,1,0}));
        }
    }
    if (r0 > 1e-4f) {
        uint32_t c = addVert(base, {0, -1, 0});
        for (int i = 0; i < seg; i++) {
            float a0 = TAU * i / seg, a1 = TAU * (i + 1) / seg;
            tri(c, addVert(base + v3{std::cos(a0)*r0, 0, std::sin(a0)*r0}, {0,-1,0}),
                   addVert(base + v3{std::cos(a1)*r0, 0, std::sin(a1)*r0}, {0,-1,0}));
        }
    }
}

void MeshBuilder::sphere(v3 c, float r, int seg, int rings) {
    for (int y = 0; y < rings; y++) {
        float t0 = PI * y / rings, t1 = PI * (y + 1) / rings;
        for (int x = 0; x < seg; x++) {
            float p0 = TAU * x / seg, p1 = TAU * (x + 1) / seg;
            auto sp = [&](float th, float ph) {
                return v3{std::sin(th)*std::cos(ph), std::cos(th), std::sin(th)*std::sin(ph)};
            };
            v3 n00 = sp(t0,p0), n10 = sp(t0,p1), n01 = sp(t1,p0), n11 = sp(t1,p1);
            quad(addVert(c + n00*r, n00), addVert(c + n10*r, n10),
                 addVert(c + n11*r, n11), addVert(c + n01*r, n01));
        }
    }
}

void MeshBuilder::jitterSphere(v3 c, float r, int seg, int rings, uint32_t seed, float amount) {
    auto rad = [&](float th, float ph) {
        float x = std::sin(th)*std::cos(ph), y = std::cos(th), z = std::sin(th)*std::sin(ph);
        float n = fbm(x*2.1f + seed*7.3f, z*2.1f + y*1.7f + seed*3.1f, 3);
        return r * (1.0f + (n - 0.5f) * 2.0f * amount);
    };
    for (int y = 0; y < rings; y++) {
        float t0 = PI * y / rings, t1 = PI * (y + 1) / rings;
        for (int x = 0; x < seg; x++) {
            float p0 = TAU * x / seg, p1 = TAU * (x + 1) / seg;
            auto sp = [&](float th, float ph) {
                v3 d{std::sin(th)*std::cos(ph), std::cos(th), std::sin(th)*std::sin(ph)};
                return c + d * rad(th, ph);
            };
            v3 a = sp(t0,p0), b = sp(t1,p0), cc = sp(t1,p1), d = sp(t0,p1);
            // Displacement makes these quads strongly non-planar, so each
            // triangle gets its own normal and is oriented away from the centre.
            auto emitTri = [&](v3 A, v3 B, v3 C) {
                v3 n = cross(B - A, C - A);
                if (dot(n, A - c) < 0.0f) { std::swap(B, C); n = cross(B - A, C - A); }
                if (length2(n) < 1e-12f) return;
                n = normalize(n);
                tri(addVert(A, n), addVert(B, n), addVert(C, n));
            };
            emitTri(a, d, cc);
            emitTri(a, cc, b);
        }
    }
}

void MeshBuilder::ringFlat(v3 c, float rIn, float rOut, int seg) {
    for (int i = 0; i < seg; i++) {
        float a0 = TAU * i / seg, a1 = TAU * (i + 1) / seg;
        v3 d0{std::cos(a0), 0, std::sin(a0)}, d1{std::cos(a1), 0, std::sin(a1)};
        quad(addVert(c + d0*rIn,  {0,1,0}), addVert(c + d1*rIn,  {0,1,0}),
             addVert(c + d1*rOut, {0,1,0}), addVert(c + d0*rOut, {0,1,0}));
    }
}

// ---------------------------------------------------------------- palette
static const v3 ARMOR   {0.70f, 0.69f, 0.64f};
static const v3 ARMOR_D {0.46f, 0.46f, 0.44f};
static const v3 DARK    {0.15f, 0.16f, 0.18f};
static const v3 STEEL   {0.55f, 0.57f, 0.61f};
static const v3 GLOW_W  {1.00f, 0.82f, 0.45f};
static const v3 CRYSTAL {0.32f, 0.78f, 0.95f};

// ---------------------------------------------------------------- models
static void buildWorker(MeshBuilder& b) {
    b.mat(DARK, 0.92f); // tracks
    b.box({ 0.58f, 0.24f, 0.0f}, {0.17f, 0.24f, 0.82f});
    b.box({-0.58f, 0.24f, 0.0f}, {0.17f, 0.24f, 0.82f});
    b.mat(ARMOR_D, 0.6f, 0.25f);
    b.box({0, 0.62f, 0}, {0.52f, 0.20f, 0.76f});          // chassis
    b.mat(ARMOR, 0.55f, 0.3f);
    b.wedge({0, 1.00f, -0.05f}, {0.46f, 0.26f, 0.62f}, 0.22f, 0.18f);
    b.mat(ARMOR, 0.5f, 0.2f, 1.0f);                        // team-coloured cab
    b.box({0, 1.30f, 0.16f}, {0.34f, 0.22f, 0.38f});
    b.mat(GLOW_W, 0.25f, 0.0f, 0.0f, 2.6f);                // cab window
    b.box({0, 1.32f, 0.55f}, {0.25f, 0.13f, 0.04f});
    b.mat(STEEL, 0.45f, 0.7f);                             // fusion cutter arms
    b.box({ 0.46f, 0.78f, 0.72f}, {0.09f, 0.09f, 0.34f});
    b.box({-0.46f, 0.78f, 0.72f}, {0.09f, 0.09f, 0.34f});
    b.mat(GLOW_W, 0.3f, 0.0f, 0.0f, 1.4f);
    b.box({ 0.46f, 0.78f, 1.06f}, {0.05f, 0.05f, 0.10f});
    b.box({-0.46f, 0.78f, 1.06f}, {0.05f, 0.05f, 0.10f});
    b.mat(DARK, 0.8f);
    b.cyl({0.3f, 1.5f, -0.3f}, 0.03f, 0.02f, 0.5f, 5);     // antenna
}

static void buildTrooper(MeshBuilder& b) {
    b.mat(ARMOR_D, 0.7f, 0.1f);                            // legs
    b.box({ 0.26f, 0.44f,  0.06f}, {0.17f, 0.44f, 0.20f});
    b.box({-0.26f, 0.44f, -0.06f}, {0.17f, 0.44f, 0.20f});
    b.mat(DARK, 0.85f);                                    // boots
    b.box({ 0.26f, 0.08f,  0.10f}, {0.19f, 0.09f, 0.26f});
    b.box({-0.26f, 0.08f, -0.02f}, {0.19f, 0.09f, 0.26f});
    b.mat(ARMOR, 0.55f, 0.15f);                            // torso
    b.wedge({0, 1.28f, 0}, {0.40f, 0.42f, 0.26f}, 0.15f, 0.10f);
    b.mat(ARMOR, 0.5f, 0.2f, 1.0f);                        // pauldrons
    b.box({ 0.50f, 1.56f, 0}, {0.16f, 0.19f, 0.25f});
    b.box({-0.50f, 1.56f, 0}, {0.16f, 0.19f, 0.25f});
    b.mat(DARK, 0.75f);                                    // backpack
    b.box({0, 1.36f, -0.35f}, {0.28f, 0.30f, 0.13f});
    b.mat(GLOW_W, 0.3f, 0.0f, 0.0f, 1.8f);
    b.box({0, 1.62f, -0.44f}, {0.16f, 0.05f, 0.04f});
    b.mat(ARMOR, 0.5f, 0.2f);                              // helmet
    b.sphere({0, 1.90f, 0.02f}, 0.26f, 10, 7);
    b.mat(v3{0.25f, 0.85f, 1.0f}, 0.15f, 0.0f, 0.0f, 2.2f);// visor
    b.box({0, 1.90f, 0.22f}, {0.15f, 0.07f, 0.08f});
    b.mat(DARK, 0.6f, 0.4f);                               // gauss rifle
    b.box({0.40f, 1.20f, 0.36f}, {0.09f, 0.10f, 0.36f});
    b.cyl({0.40f, 1.24f, 0.70f}, 0.055f, 0.05f, 0.0f, 6);
    b.push(translate({0.40f, 1.24f, 0.70f}) * rotateX(PI * 0.5f));
    b.cyl({0, 0, 0}, 0.055f, 0.05f, 0.42f, 6);
    b.pop();
}

static void buildMaulerHull(MeshBuilder& b) {
    b.mat(DARK, 0.92f);                                    // tracks
    b.box({ 1.32f, 0.40f, 0}, {0.34f, 0.40f, 1.95f});
    b.box({-1.32f, 0.40f, 0}, {0.34f, 0.40f, 1.95f});
    b.mat(STEEL, 0.7f, 0.5f);                              // road wheels
    for (int i = -2; i <= 2; i++) {
        b.push(translate({1.32f, 0.40f, i * 0.78f}) * rotateZ(PI * 0.5f));
        b.cyl({0, -0.36f, 0}, 0.26f, 0.26f, 0.72f, 8); b.pop();
        b.push(translate({-1.32f, 0.40f, i * 0.78f}) * rotateZ(PI * 0.5f));
        b.cyl({0, -0.36f, 0}, 0.26f, 0.26f, 0.72f, 8); b.pop();
    }
    b.mat(ARMOR_D, 0.6f, 0.35f);                           // lower hull
    b.box({0, 0.72f, 0}, {1.20f, 0.30f, 1.85f});
    b.mat(ARMOR, 0.55f, 0.3f);                             // upper hull + glacis
    b.wedge({0, 1.14f, -0.15f}, {1.15f, 0.30f, 1.70f}, 0.20f, 0.12f);
    b.mat(ARMOR, 0.5f, 0.25f, 1.0f);                       // team side skirts
    b.box({ 1.19f, 1.05f, 0.2f}, {0.06f, 0.20f, 1.35f});
    b.box({-1.19f, 1.05f, 0.2f}, {0.06f, 0.20f, 1.35f});
    b.mat(GLOW_W, 0.3f, 0.0f, 0.0f, 1.6f);                 // running lights
    b.box({ 0.85f, 1.30f, 1.56f}, {0.10f, 0.05f, 0.04f});
    b.box({-0.85f, 1.30f, 1.56f}, {0.10f, 0.05f, 0.04f});
}

// Turret is a separate mesh so it can be aimed independently of the hull.
static void buildMaulerTurret(MeshBuilder& b) {
    b.mat(ARMOR, 0.55f, 0.3f);
    b.wedge({0, 0.30f, -0.15f}, {0.82f, 0.30f, 0.95f}, 0.25f, 0.20f);
    b.mat(ARMOR_D, 0.6f, 0.35f);                           // mantlet
    b.box({0, 0.30f, 0.80f}, {0.34f, 0.24f, 0.22f});
    b.mat(STEEL, 0.42f, 0.75f);                            // 120mm barrel
    b.push(translate({0, 0.30f, 0.95f}) * rotateX(PI * 0.5f));
    b.cyl({0, 0, 0}, 0.135f, 0.115f, 1.95f, 10);
    b.cyl({0, 1.72f, 0}, 0.185f, 0.185f, 0.26f, 10);       // muzzle brake
    b.pop();
    b.mat(ARMOR, 0.5f, 0.25f, 1.0f);                       // team band
    b.box({0, 0.58f, -0.5f}, {0.70f, 0.06f, 0.30f});
    b.mat(DARK, 0.7f);                                     // commander hatch
    b.cyl({0.32f, 0.58f, -0.3f}, 0.22f, 0.22f, 0.10f, 8);
}

static void buildFoundry(MeshBuilder& b) {
    b.mat(ARMOR_D, 0.7f, 0.2f);
    b.cyl({0, 0, 0}, 4.6f, 4.3f, 1.10f, 8);                // foundation
    b.mat(ARMOR, 0.58f, 0.25f);
    b.cyl({0, 1.10f, 0}, 3.7f, 3.5f, 2.40f, 8);            // main drum
    b.mat(GLOW_W, 0.25f, 0.0f, 0.0f, 2.2f);                // window band
    b.cyl({0, 2.05f, 0}, 3.56f, 3.56f, 0.42f, 8, false);
    b.mat(ARMOR, 0.55f, 0.3f);
    b.cyl({0, 3.50f, 0}, 3.5f, 2.5f, 1.00f, 8);            // shoulder
    b.mat(ARMOR, 0.5f, 0.25f, 1.0f);                       // team landing ring
    b.ringFlat({0, 4.52f, 0}, 1.65f, 2.45f, 24);
    b.mat(ARMOR_D, 0.6f, 0.3f);
    b.cyl({0, 4.50f, 0}, 1.6f, 1.4f, 0.55f, 8);            // control tower
    b.mat(v3{0.30f, 0.80f, 1.0f}, 0.2f, 0.0f, 0.0f, 2.4f);
    b.cyl({0, 4.72f, 0}, 1.44f, 1.44f, 0.26f, 8, false);
    b.mat(STEEL, 0.5f, 0.6f);                              // corner pylons
    for (int i = 0; i < 4; i++) {
        float a = PI * 0.25f + i * PI * 0.5f;
        b.box({std::cos(a) * 4.15f, 1.5f, std::sin(a) * 4.15f}, {0.28f, 1.5f, 0.28f});
    }
    b.mat(GLOW_W, 0.3f, 0.0f, 0.0f, 1.5f);
    for (int i = 0; i < 4; i++) {
        float a = PI * 0.25f + i * PI * 0.5f;
        b.box({std::cos(a) * 4.15f, 3.12f, std::sin(a) * 4.15f}, {0.16f, 0.10f, 0.16f});
    }
}

static void buildGarrison(MeshBuilder& b) {
    b.mat(ARMOR_D, 0.72f, 0.2f);
    b.box({0, 0.30f, 0}, {3.20f, 0.30f, 3.70f});           // slab
    b.mat(ARMOR, 0.58f, 0.25f);
    b.wedge({0, 1.70f, -0.2f}, {2.80f, 1.10f, 3.30f}, 0.18f, 0.14f);
    b.mat(ARMOR_D, 0.65f, 0.3f);                           // roof housing
    b.box({0, 3.05f, -0.9f}, {1.90f, 0.35f, 1.70f});
    b.mat(v3{0.9f, 0.35f, 0.15f}, 0.3f, 0.0f, 0.0f, 2.0f); // bay door
    b.box({0, 1.25f, 3.02f}, {1.25f, 0.85f, 0.10f});
    b.mat(DARK, 0.8f);
    b.box({0, 1.20f, 3.14f}, {1.45f, 1.05f, 0.06f});
    b.mat(ARMOR, 0.5f, 0.25f, 1.0f);                       // team stripe
    b.box({0, 2.72f, 0.6f}, {2.72f, 0.18f, 2.55f});
    b.mat(STEEL, 0.5f, 0.7f);
    b.cyl({-2.3f, 3.4f, -2.6f}, 0.09f, 0.05f, 1.7f, 6);    // antennae
    b.cyl({ 2.3f, 3.4f, -2.6f}, 0.09f, 0.05f, 1.4f, 6);
    b.mat(GLOW_W, 0.3f, 0.0f, 0.0f, 1.8f);
    b.box({-2.3f, 5.15f, -2.6f}, {0.07f, 0.07f, 0.07f});
    b.box({ 2.3f, 4.85f, -2.6f}, {0.07f, 0.07f, 0.07f});
}

static void buildWorkshop(MeshBuilder& b) {
    b.mat(ARMOR_D, 0.72f, 0.2f);
    b.box({0, 0.32f, 0}, {3.70f, 0.32f, 4.20f});
    b.mat(ARMOR, 0.60f, 0.28f);                            // arched hangar
    b.push(translate({0, 1.15f, -3.6f}) * rotateX(PI * 0.5f));
    b.cyl({0, 0, 0}, 2.55f, 2.55f, 7.2f, 12);
    b.pop();
    b.mat(ARMOR_D, 0.62f, 0.3f);                           // end caps
    b.box({0, 1.15f, -3.75f}, {2.60f, 1.45f, 0.22f});
    b.mat(v3{0.9f, 0.35f, 0.15f}, 0.3f, 0.0f, 0.0f, 1.9f); // hangar mouth
    b.box({0, 1.05f, 3.66f}, {1.85f, 1.25f, 0.12f});
    b.mat(DARK, 0.8f);
    b.box({0, 1.20f, 3.78f}, {2.30f, 1.55f, 0.10f});
    b.mat(ARMOR, 0.5f, 0.25f, 1.0f);                       // team roof band
    b.box({0, 3.62f, -0.4f}, {1.30f, 0.16f, 3.0f});
    b.mat(STEEL, 0.55f, 0.6f);                             // exhaust stacks
    b.cyl({-2.55f, 1.6f, -2.5f}, 0.42f, 0.36f, 2.5f, 10);
    b.cyl({ 2.55f, 1.6f, -2.5f}, 0.42f, 0.36f, 2.5f, 10);
    b.mat(DARK, 0.9f);
    b.cyl({-2.55f, 4.05f, -2.5f}, 0.37f, 0.37f, 0.12f, 10);
    b.cyl({ 2.55f, 4.05f, -2.5f}, 0.37f, 0.37f, 0.12f, 10);
}

static void buildBunkhouse(MeshBuilder& b) {
    b.mat(ARMOR_D, 0.72f, 0.2f);
    b.cyl({0, 0, 0}, 2.35f, 2.20f, 0.45f, 8);
    b.mat(ARMOR, 0.58f, 0.28f);
    b.cyl({0, 0.45f, 0}, 2.05f, 1.75f, 0.95f, 8);
    b.mat(ARMOR, 0.5f, 0.25f, 1.0f);
    b.ringFlat({0, 1.42f, 0}, 1.05f, 1.72f, 16);
    b.mat(v3{0.35f, 0.85f, 1.0f}, 0.22f, 0.0f, 0.0f, 2.4f);
    b.cyl({0, 1.40f, 0}, 1.02f, 0.82f, 0.30f, 8);
    b.mat(STEEL, 0.5f, 0.6f);                              // vents
    for (int i = 0; i < 4; i++) {
        float a = PI * 0.25f + i * PI * 0.5f;
        b.box({std::cos(a) * 1.95f, 0.85f, std::sin(a) * 1.95f}, {0.22f, 0.42f, 0.22f});
    }
}

static void buildOre(MeshBuilder& b) {
    Rng rng(99);
    b.mat(CRYSTAL, 0.16f, 0.15f, 0.0f, 0.55f);
    const int n = 6;
    for (int i = 0; i < n; i++) {
        float a = TAU * i / n + rng.range(-0.3f, 0.3f);
        float d = rng.range(0.25f, 0.95f);
        float hgt = rng.range(1.1f, 2.3f);
        float lean = rng.range(0.10f, 0.30f);
        b.push(translate({std::cos(a) * d, 0, std::sin(a) * d}) * rotateZ(std::cos(a) * lean) * rotateX(-std::sin(a) * lean));
        b.cyl({0, 0, 0}, rng.range(0.22f, 0.36f), 0.03f, hgt, 5);
        b.pop();
    }
    b.mat(v3{0.20f, 0.26f, 0.32f}, 0.85f);                 // rocky base
    b.jitterSphere({0, 0.05f, 0}, 1.05f, 10, 6, 3, 0.22f);
}

static void buildRock(MeshBuilder& b) {
    b.mat(v3{0.34f, 0.32f, 0.30f}, 0.9f);
    b.jitterSphere({0, 0.0f, 0}, 1.0f, 12, 8, 7, 0.30f);
}

static void buildProjectile(MeshBuilder& b) {
    b.mat(v3{1.0f, 0.75f, 0.35f}, 0.3f, 0.0f, 0.0f, 4.0f);
    b.box({0, 0, 0}, {0.07f, 0.07f, 0.42f});
}

static void buildSelRing(MeshBuilder& b) {
    b.mat(v3{1.0f, 1.0f, 1.0f}, 0.5f, 0.0f, 1.0f, 1.0f);
    b.ringFlat({0, 0, 0}, 0.91f, 1.0f, 48);
}

// ---------------------------------------------------------------- assembly
static void emitMesh(std::vector<MeshVertex>& V, std::vector<uint32_t>& I,
                     MeshRange& R, MeshBuilder& b) {
    R.baseVertex = (int32_t)V.size();
    R.firstIndex = (uint32_t)I.size();
    R.indexCount = (uint32_t)b.idx.size();
    float rad = 0.0f, hi = 0.0f;
    for (const auto& v : b.verts) {
        rad = std::max(rad, std::sqrt(v.px * v.px + v.pz * v.pz));
        hi  = std::max(hi, v.py);
    }
    R.radius = rad; R.height = hi;
    V.insert(V.end(), b.verts.begin(), b.verts.end());
    I.insert(I.end(), b.idx.begin(), b.idx.end());
}

void buildMeshLibrary(std::vector<MeshVertex>& V, std::vector<uint32_t>& I,
                      MeshRange R[MESH_COUNT]) {
    V.clear(); I.clear();
    using Fn = void (*)(MeshBuilder&);
    const Fn fns[MESH_COUNT] = {
        buildWorker, buildTrooper, buildMaulerHull, buildMaulerTurret,
        buildFoundry, buildGarrison, buildWorkshop, buildBunkhouse,
        buildOre, buildRock, buildProjectile, buildSelRing
    };
    for (int i = 0; i < MESH_COUNT; i++) {
        MeshBuilder b;
        fns[i](b);
        emitMesh(V, I, R[i], b);
    }
}

} // namespace sf
