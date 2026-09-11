// RenderTypes.h — POD layouts shared between the C++ core and the Metal shaders.
// Every struct is a multiple of 16 bytes so it maps 1:1 onto MSL declarations.
#pragma once
#include "../core/Math.h"

namespace sf {

// Vertex for every procedurally generated object mesh.
struct MeshVertex {
    float px, py, pz, _p0;
    float nx, ny, nz, _p1;
    float cr, cg, cb, rough;   // base albedo + roughness
    float metal, team, emis, ao;  // team = blend weight toward the faction colour
};
static_assert(sizeof(MeshVertex) == 64, "MeshVertex layout");

// One drawn object. Instanced; the renderer sorts these by mesh id.
struct InstanceData {
    m4 model;
    v4 tint;    // rgb multiplier + roughness bias
    v4 team;    // faction rgb + metallic bias
    v4 fx;      // x=emissive gain, y=damage flash, z=alpha, w=build-in progress
};
static_assert(sizeof(InstanceData) == 112, "InstanceData layout");

// World-space camera-facing sprite: particles, ground decals, selection glow.
struct Billboard {
    v3 pos; float size;
    v4 color;             // rgba, premultiplied by the emitter
    // kind: 0 soft puff, 1 spark, 2 ring, 3 shockwave, 4 explosion sheet,
    // 5 ground scorch, 6 smoke plume. `param` is a per-kind extra -- for the
    // scorch it picks which of the four marks on the sheet.
    float rot, kind, fade, param;
};
static_assert(sizeof(Billboard) == 48, "Billboard layout");

// Screen-space (pixel) vertex for HUD, text and health bars.
struct UIVertex {
    float x, y, u, v;
    float r, g, b, a;
};
static_assert(sizeof(UIVertex) == 32, "UIVertex layout");

// Per-frame constants bound to every pass.
struct FrameUniforms {
    m4 viewProj;
    m4 view;
    m4 lightViewProj;
    m4 invViewProj;   // reconstructs world-space view rays in the sky pass
    v4 camPos;      // xyz, w = time seconds
    v4 sunDir;      // xyz toward the sun, w = intensity
    v4 sunColor;    // rgb, w = ambient intensity
    v4 fog;         // rgb fog/sky tint, w = density
    v4 misc;        // x = water level, y = shadow texel size, z = nearZ, w = farZ
    v4 texParams;   // x = 1/mapSize, y = fog-of-war strength, z = water normal strength
    // Which of the optional sheets actually loaded, 0 or 1 each. The shaders keep
    // a procedural fallback for every one of them, so a missing assets/ directory
    // still renders -- just with the noise the textures replaced.
    v4 texFlags;    // x = clouds, y = particle sheet, z = terrain macro variation
};
static_assert(sizeof(FrameUniforms) == 368, "FrameUniforms layout");

// Identifies a mesh range inside the renderer's shared vertex/index buffers.
enum MeshId : int {
    MESH_WORKER = 0,
    MESH_TROOPER,
    MESH_MAULER_HULL,
    MESH_MAULER_TURRET,
    MESH_FOUNDRY,
    MESH_GARRISON,
    MESH_WORKSHOP,
    MESH_BUNKHOUSE,
    MESH_ORE,
    MESH_BOULDER,
    MESH_PROJECTILE,
    MESH_SEL_RING,
    MESH_COUNT
};

} // namespace sf
