// Shaders.h — the entire Metal Shading Language source, compiled at launch via
// newLibraryWithSource. Kept as one translation unit so helpers are shared.
#pragma once

namespace sf {

inline const char* kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- shared types
struct FrameUniforms {
    float4x4 viewProj;
    float4x4 view;
    float4x4 lightViewProj;
    float4x4 invViewProj;
    float4 camPos;     // xyz, w = time
    float4 sunDir;     // xyz toward sun, w = intensity
    float4 sunColor;   // rgb, w = ambient
    float4 fog;        // rgb, w = density
    float4 misc;       // x = waterLevel, y = shadow texel, z = nearZ, w = farZ
    float4 texParams;  // x = 1/mapSize, y = fog-of-war strength, z = water normal strength
    float4 texFlags;   // x = clouds, y = particle sheet, z = terrain macro (0 or 1)
};

struct TVert  { packed_float3 pos; packed_float3 nrm; float ao; float var; };
struct MVert  { packed_float3 pos; float _p0; packed_float3 nrm; float _p1;
                float4 colRough; float4 mte; };
struct Instance { float4x4 model; float4 tint; float4 team; float4 fx; };
struct BBoard { packed_float3 pos; float size; float4 color; float rot, kind, fade, param; };
struct UIVert { float2 xy; float2 uv; float4 rgba; };

// ---------------------------------------------------------------- fog of war
constexpr sampler fowSmp(filter::linear, mip_filter::none, address::clamp_to_edge);

// The player's own fog of war. r = currently observed, g = ever observed; both
// come from the same 64x64 grid the simulation already keeps per team, eased on
// the CPU and stretched over the map by the hardware's bilinear filter.
static inline float2 fowAt(texture2d<float> fowTex, float3 wpos,
                           constant FrameUniforms& U) {
    return fowTex.sample(fowSmp, wpos.xz * U.texParams.x).rg;
}
// Ground that was scouted but is no longer watched stays legible as a dim,
// desaturated memory -- that is the shape of the terrain the player remembers,
// not live information. Ground never seen at all goes nearly black.
static inline float3 applyFOW(float3 col, float3 wpos, texture2d<float> fowTex,
                              constant FrameUniforms& U) {
    float s = U.texParams.y;
    if (s < 0.001) return col;
    float2 f = fowAt(fowTex, wpos, U);
    float lum = dot(col, float3(0.2126, 0.7152, 0.0722));
    float3 memory = mix(float3(lum), col, 0.30) * 0.40;
    float3 out = mix(mix(memory, col, f.r), float3(0.006, 0.008, 0.013), 1.0 - f.g);
    return mix(col, out, s);
}

// ---------------------------------------------------------------- noise
static inline float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
static inline float vnoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + float2(1, 0));
    float c = hash21(i + float2(0, 1)), d = hash21(i + float2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
static inline float fbm2(float2 p, int oct) {
    float s = 0.0, a = 0.5, n = 0.0;
    for (int i = 0; i < oct; i++) { s += a * vnoise(p); n += a; p *= 2.03; a *= 0.5; }
    return s / n;
}

// ---------------------------------------------------------------- materials
constant int MAT_GROUND  = 0;
constant int MAT_CLIFF   = 1;
constant int MAT_ARMOR   = 2;
constant int MAT_CRYSTAL = 3;

// Perturb a world normal by a tangent-space sample without a real TBN. The
// meshes have no UVs and the terrain is a height field, so a basis derived from
// the geometric normal is both sufficient and much cheaper than storing tangents.
static inline float3 applyDetailNormal(float3 N, float2 nxy, float strength) {
    float3 up = (abs(N.y) > 0.95) ? float3(0, 0, 1) : float3(0, 1, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    return normalize(N + (T * nxy.x + B * nxy.y) * strength);
}

// ---------------------------------------------------------------- lighting
// Trowbridge-Reitz GGX with Smith visibility and a Schlick Fresnel term.
static inline float3 pbrDirect(float3 N, float3 V, float3 L, float3 albedo,
                               float rough, float metallic, float3 lightCol) {
    float3 H = normalize(V + L);
    float NdL = max(dot(N, L), 0.0);
    float NdV = max(dot(N, V), 1e-4);
    float NdH = max(dot(N, H), 0.0);
    float VdH = max(dot(V, H), 0.0);
    float a = max(rough * rough, 0.008);
    float a2 = a * a;
    float d = NdH * NdH * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159265 * d * d, 1e-6);
    float k = a * 0.5;
    float G = (NdL / (NdL * (1.0 - k) + k)) * (NdV / (NdV * (1.0 - k) + k));
    float3 F0 = mix(float3(0.04), albedo, metallic);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdH, 5.0);
    float3 spec = D * G * F / (4.0 * NdL * NdV + 1e-4);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    return (kd * albedo / 3.14159265 + spec) * lightCol * NdL;
}

// Two-colour hemisphere ambient: sky above, bounced ground light below.
static inline float3 ambientIBL(float3 N, float3 albedo, float ao, float amb, float3 skyTint) {
    float3 sky = skyTint * 1.10;
    float3 grd = float3(0.20, 0.18, 0.15);
    float3 env = mix(grd, sky, N.y * 0.5 + 0.5);
    return albedo * env * amb * ao;
}

static inline float sampleShadow(depth2d<float> shadowMap, sampler shadowSmp,
                                 float4 lightClip, float NdL, float texel) {
    float3 p = lightClip.xyz / lightClip.w;
    float2 uv = float2(p.x * 0.5 + 0.5, -p.y * 0.5 + 0.5);
    if (uv.x < 0.002 || uv.x > 0.998 || uv.y < 0.002 || uv.y > 0.998 || p.z > 1.0) return 1.0;
    // Slope-scaled bias: grazing light needs more offset to avoid acne.
    float bias = mix(0.0022, 0.0005, saturate(NdL));
    float ref = p.z - bias;
    // Five taps, not nine. The hardware comparison sampler already gives 2x2
    // bilinear PCF per tap, so a rotated cross covers the same footprint for
    // little visible difference -- and this is the scene pass's biggest
    // bandwidth consumer on an integrated GPU.
    const float2 kOff[5] = { float2(0, 0), float2(1.1, 0.4), float2(-0.4, 1.1),
                             float2(-1.1, -0.4), float2(0.4, -1.1) };
    float sum = 0.0;
    for (int i = 0; i < 5; i++)
        sum += shadowMap.sample_compare(shadowSmp, uv + kOff[i] * texel * 1.6, ref);
    return sum / 5.0;
}

// Height-and-distance fog keeps far terrain from reading as a flat cutout.
static inline float3 applyFog(float3 color, float3 worldPos, constant FrameUniforms& U) {
    float3 toCam = U.camPos.xyz - worldPos;
    float dist = length(toCam);
    float heightFalloff = exp(-max(worldPos.y - 2.0, 0.0) * 0.018);
    float f = 1.0 - exp(-dist * U.fog.w * heightFalloff);
    f = saturate(f);
    // Sun-facing scattering warms the fog near the light direction.
    float sunAmt = saturate(dot(normalize(-toCam), normalize(U.sunDir.xyz)));
    float3 fogCol = mix(U.fog.rgb, U.sunColor.rgb * 0.9, pow(sunAmt, 5.0) * 0.55);
    return mix(color, fogCol, f);
}

// ---------------------------------------------------------------- sky
struct SkyOut { float4 pos [[position]]; float2 ndc; };

vertex SkyOut skyVS(uint vid [[vertex_id]]) {
    // Oversized triangle covering the viewport.
    float2 p = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    SkyOut o;
    o.pos = float4(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

static inline float3 skyColor(float3 dir, constant FrameUniforms& U,
                              texture2d<float> cloudTex, sampler smp) {
    float3 sun = normalize(U.sunDir.xyz);
    float up = saturate(dir.y * 0.5 + 0.5);
    float3 horizon = U.fog.rgb;
    float3 zenith  = float3(0.16, 0.28, 0.52);
    float3 col = mix(horizon, zenith, pow(saturate(dir.y), 0.55));
    // Sun disc plus a broad forward-scatter halo.
    float sd = saturate(dot(dir, sun));
    col += U.sunColor.rgb * pow(sd, 900.0) * 12.0;
    col += U.sunColor.rgb * pow(sd, 12.0) * 0.28;
    col += U.sunColor.rgb * pow(sd, 3.0) * 0.06;
    // Layered cloud deck projected onto the upper hemisphere. Two samples of one
    // tiling density map, drifting at different rates, stand in for the nine
    // octaves of value noise this used to cost. That mattered more than it looks:
    // the sky pass runs full-screen with the depth test off, so the noise was
    // shading every pixel the terrain later covered, and waterFS calls this
    // function a second time for reflections.
    if (dir.y > 0.01) {
        float2 cp = dir.xz / max(dir.y, 0.06) * 0.22;
        float t = U.camPos.w * 0.004;
        float c, c2;
        if (U.texFlags.x > 0.5) {
            // Explicit LOD, not hardware-derived. This projection divides by
            // dir.y, so approaching the horizon the UV derivative explodes and
            // neighbouring pixels land on wildly different mips -- which thrashes
            // the texture cache badly enough that two samples measured *slower*
            // than the nine octaves of value noise they replaced. The scale grows
            // as 1/dir.y^2, so the mip level is just its log.
            float lod = clamp(-2.0 * log2(max(dir.y, 0.02)), 0.0, 9.0);
            c  = cloudTex.sample(smp, cp * 0.35 + float2(t, t * 0.4), level(lod)).r;
            // Rotated as well as rescaled: a second sample of the same map at a
            // plain multiple of the first scale lines its features up with them.
            c2 = cloudTex.sample(smp, float2(cp.y, -cp.x) * 0.81
                                      - float2(t * 1.7, 0.0), level(lod)).r;
        } else {
            c  = fbm2(cp + float2(t, t * 0.4), 5);
            c2 = fbm2(cp * 2.3 - float2(t * 1.7, 0.0), 4);
        }
        // The map's own histogram sits lower than fBm's, so the coverage
        // threshold is not the same number as the procedural path's.
        float thr = (U.texFlags.x > 0.5) ? 0.40 : 0.46;
        float cov = saturate((c * 0.7 + c2 * 0.3 - thr) * 3.3);
        cov *= smoothstep(0.02, 0.30, dir.y);
        float3 lit = mix(float3(0.62, 0.64, 0.70), U.sunColor.rgb * 1.25, pow(sd, 4.0) * 0.7 + 0.25);
        col = mix(col, lit, cov * 0.82);
    }
    return col * (0.55 + 0.45 * up);
}

fragment float4 skyFS(SkyOut in [[stage_in]], constant FrameUniforms& U [[buffer(0)]],
                      texture2d<float> cloudTex [[texture(8)]],
                      sampler matSmp [[sampler(1)]]) {
    float4 far = U.invViewProj * float4(in.ndc, 1.0, 1.0);
    float4 near = U.invViewProj * float4(in.ndc, 0.0, 1.0);
    float3 dir = normalize(far.xyz / far.w - near.xyz / near.w);
    return float4(skyColor(dir, U, cloudTex, matSmp), 1.0);
}
)MSL";

} // namespace sf

namespace sf {

inline const char* kShaderSource2 = R"MSL(
// ---------------------------------------------------------------- terrain
struct TerrainOut {
    float4 pos [[position]];
    float3 wpos;
    float3 nrm;
    float2 aovar;
    float4 lightClip;
};

vertex TerrainOut terrainVS(uint vid [[vertex_id]],
                            device const TVert* verts [[buffer(0)]],
                            constant FrameUniforms& U [[buffer(1)]]) {
    TVert v = verts[vid];
    float3 p = float3(v.pos);
    TerrainOut o;
    o.pos = U.viewProj * float4(p, 1.0);
    o.wpos = p;
    o.nrm = float3(v.nrm);
    o.aovar = float2(v.ao, v.var);
    o.lightClip = U.lightViewProj * float4(p, 1.0);
    return o;
}

fragment float4 terrainFS(TerrainOut in [[stage_in]],
                          constant FrameUniforms& U [[buffer(1)]],
                          depth2d<float> shadowMap [[texture(0)]],
                          texture2d_array<float> matAlbedo [[texture(1)]],
                          texture2d_array<float> matNormal [[texture(2)]],
                          texture2d<float> fowTex [[texture(4)]],
                          texture2d<float> macroTex [[texture(10)]],
                          sampler shadowSmp [[sampler(0)]],
                          sampler matSmp [[sampler(1)]]) {
    float3 N = normalize(in.nrm);
    float2 uv = in.wpos.xz;
    float h = in.wpos.y;
    float slope = N.y;               // 1 = flat ground, 0 = vertical cliff
    float macro = in.aovar.y;

    // Biome hue variation. These two only ever drive colour mixes -- surface
    // detail comes from the material textures below -- so the tiling map trades
    // the old sub-metre noise for patchiness at a scale fBm was bad at, which is
    // the more useful half. Sampled rotated as well as rescaled so the two do not
    // line up, and at rates low enough that the tile seam does not read as a grid.
    float d1, d2;
    if (U.texFlags.z > 0.5) {
        d1 = macroTex.sample(matSmp, uv * 0.030).r;
        d2 = macroTex.sample(matSmp, float2(uv.y, -uv.x) * 0.135 + 0.37).r;
    } else {
        d1 = fbm2(uv * 0.33, 3);
        d2 = fbm2(uv * 1.70, 3);
    }
    // Sedimentary banding used to be a third fBm here; the cliff texture supplies
    // it now, so this reuses an octave already computed above.
    float strat = d2;

    float cliff = 1.0 - smoothstep(0.50, 0.80, slope);
    float shore = 1.0 - smoothstep(U.misc.x - 0.5, U.misc.x + 2.8, h);
    float alpine = smoothstep(23.0, 31.0, h);

    float3 cGrass = mix(float3(0.145, 0.225, 0.115), float3(0.255, 0.335, 0.150), d1);
    float3 cDirt  = mix(float3(0.290, 0.232, 0.150), float3(0.395, 0.325, 0.215), d2);
    float3 cRock  = mix(float3(0.175, 0.170, 0.165), float3(0.350, 0.330, 0.305), strat);
    float3 cSand  = mix(float3(0.470, 0.425, 0.305), float3(0.560, 0.510, 0.380), d2);
    float3 cAlp   = mix(float3(0.295, 0.305, 0.320), float3(0.430, 0.440, 0.455), d2);

    float grassAmt = saturate(smoothstep(0.60, 0.87, slope) * (1.0 - alpine) * (0.45 + 0.80 * macro));
    float3 albedo = cDirt;
    albedo = mix(albedo, cGrass, grassAmt);
    albedo = mix(albedo, cAlp,  alpine * smoothstep(0.45, 0.88, slope));
    albedo = mix(albedo, cRock, cliff);
    albedo = mix(albedo, cSand, shore * (1.0 - cliff) * 0.9);

    float rough = mix(0.93, 0.82, cliff);
    rough = mix(rough, 0.70, shore * 0.6);   // damp sand near the shoreline

    // Perturb the normal with the gradient of a detail octave so flat plateaus
    // still catch the light instead of reading as untextured polygons.
    // --- material textures -------------------------------------------------
    // Replaces the three-tap fBm gradient that used to stand in for surface
    // detail. Ground projects flat on XZ; a cliff face is near-vertical, where a
    // flat projection would smear, so it uses the cheap cousin of triplanar --
    // pick the dominant horizontal axis and map world height to V. That also
    // lines the rock's strata up with the terrain's own terracing.
    float distFade = 1.0 - smoothstep(60.0, 230.0, length(U.camPos.xyz - in.wpos));
    if (U.misc.z > 0.001) {
        float2 gUV = uv * (1.0 / 11.0);
        float2 cUV = float2((abs(N.x) > abs(N.z)) ? in.wpos.z : in.wpos.x,
                            in.wpos.y) * (1.0 / 13.0);

        float4 gA = matAlbedo.sample(matSmp, gUV, MAT_GROUND);
        float4 cA = matAlbedo.sample(matSmp, cUV, MAT_CLIFF);
        float4 tA = mix(gA, cA, cliff);
        float4 tN = mix(matNormal.sample(matSmp, gUV, MAT_GROUND),
                        matNormal.sample(matSmp, cUV, MAT_CLIFF), cliff);

        // Modulate rather than replace: the procedural biome blend still drives
        // hue, the photograph supplies detail. The textures are pre-scaled at load
        // so their mean linear luminance is 0.5, hence the factor of two.
        float3 detail = tA.rgb * 2.0;
        albedo *= mix(float3(1.0), detail, 0.75);
        rough = saturate(rough - (tA.r - 0.5) * 0.22);
        N = applyDetailNormal(N, tN.rg * 2.0 - 1.0, (1.15 - 0.30 * cliff) * distFade);
    } else {
        // Procedural fallback when assets/ is missing.
        const float e = 0.55;
        float2 duv = uv * 0.85;
        float n0 = fbm2(duv, 2);
        float gx = fbm2(duv + float2(e, 0), 2) - n0;
        float gz = fbm2(duv + float2(0, e), 2) - n0;
        N = normalize(N + float3(-gx, 0.0, -gz) * (0.90 - 0.35 * cliff) * distFade);
    }

    float3 V = normalize(U.camPos.xyz - in.wpos);
    float3 L = normalize(U.sunDir.xyz);
    float NdL = dot(N, L);
    float shadow = sampleShadow(shadowMap, shadowSmp, in.lightClip, NdL, U.misc.y);
    float ao = in.aovar.x;

    float3 col = pbrDirect(N, V, L, albedo, rough, 0.0, U.sunColor.rgb * U.sunDir.w) * shadow;
    col += ambientIBL(N, albedo, ao, U.sunColor.w, U.fog.rgb);
    return float4(applyFOW(applyFog(col, in.wpos, U), in.wpos, fowTex, U), 1.0);
}

// ---------------------------------------------------------------- water
struct WaterOut {
    float4 pos [[position]];
    float3 wpos;
    float depth;     // metres of water above the seabed at this vertex
};

vertex WaterOut waterVS(uint vid [[vertex_id]],
                        device const TVert* verts [[buffer(0)]],
                        constant FrameUniforms& U [[buffer(1)]]) {
    TVert v = verts[vid];
    float3 p = float3(v.pos);
    // Gentle swell so the surface is never perfectly planar.
    p.y += sin(p.x * 0.22 + U.camPos.w * 1.1) * 0.045
         + sin(p.z * 0.31 - U.camPos.w * 0.8) * 0.035;
    WaterOut o;
    o.pos = U.viewProj * float4(p, 1.0);
    o.wpos = p;
    o.depth = v.ao;
    return o;
}

fragment float4 waterFS(WaterOut in [[stage_in]],
                        constant FrameUniforms& U [[buffer(1)]],
                        texture2d<float> fowTex [[texture(4)]],
                        texture2d<float> waterNrm [[texture(7)]],
                        texture2d<float> cloudTex [[texture(8)]],
                        sampler matSmp [[sampler(1)]]) {
    float t = U.camPos.w;
    float2 uv = in.wpos.xz;
    float3 N;
    float detail;            // 0..1 surface height, used to break up the foam
    if (U.texParams.z > 0.001) {
        // Two counter-scrolling samples of the ripple map. This replaces ten
        // value-noise lookups per pixel -- the old gradient needed four fBm
        // evaluations of two and three octaves each -- with two texture fetches
        // that the cache serves almost for free.
        float2 uvA = uv * 0.085 + float2(t * 0.013, t * 0.007);
        float2 uvB = uv * 0.037 - float2(t * 0.008, t * 0.015);
        float4 sA = waterNrm.sample(matSmp, uvA);
        float4 sB = waterNrm.sample(matSmp, uvB);
        // The Sobel pass stored the negated height gradient, so z reconstructs
        // from the other two and the whole thing is already a unit normal.
        float2 g = ((sA.rg * 2.0 - 1.0) * 0.62 + (sB.rg * 2.0 - 1.0) * 0.38) * U.texParams.z;
        N = normalize(float3(g.x, sqrt(saturate(1.0 - dot(g, g))), g.y));
        detail = sA.b * 0.6 + sB.b * 0.4;
    } else {
        // Procedural fallback when assets/ is missing.
        const float e = 0.35;
        float2 uvA = uv * 0.55 + float2(t * 0.35, t * 0.18);
        float2 uvB = uv * 1.30 - float2(t * 0.22, t * 0.41);
        float gx = (fbm2(uvA + float2(e,0), 3) - fbm2(uvA - float2(e,0), 3)) * 0.65
                 + (fbm2(uvB + float2(e,0), 2) - fbm2(uvB - float2(e,0), 2)) * 0.35;
        float gz = (fbm2(uvA + float2(0,e), 3) - fbm2(uvA - float2(0,e), 3)) * 0.65
                 + (fbm2(uvB + float2(0,e), 2) - fbm2(uvB - float2(0,e), 2)) * 0.35;
        N = normalize(float3(-gx * 2.4, 1.0, -gz * 2.4));
        detail = fbm2(uv * 2.6 + float2(t * 0.3, 0.0), 3);
    }

    float3 V = normalize(U.camPos.xyz - in.wpos);
    float3 L = normalize(U.sunDir.xyz);
    float fres = 0.02 + 0.98 * pow(1.0 - saturate(dot(N, V)), 5.0);

    float3 refl = skyColor(normalize(reflect(-V, N)), U, cloudTex, matSmp);
    float3 deepC    = float3(0.020, 0.075, 0.105);
    float3 shallowC = float3(0.075, 0.230, 0.245);
    float3 body = mix(shallowC, deepC, saturate(in.depth / 5.0));

    float3 col = mix(body, refl, saturate(fres * 1.05));
    // Sharp specular glint from the sun on the perturbed surface.
    float3 H = normalize(V + L);
    col += U.sunColor.rgb * pow(max(dot(N, H), 0.0), 420.0) * 5.5;

    // Foam where the water meets land, broken up with noise so it isn't a band.
    float foam = (1.0 - smoothstep(0.0, 0.85, in.depth))
               * smoothstep(0.35, 0.75, detail);
    col = mix(col, float3(0.80, 0.86, 0.88), saturate(foam) * 0.85);

    float alpha = saturate(0.42 + fres * 0.55 + saturate(in.depth / 2.2) * 0.30 + foam * 0.5);
    return float4(applyFOW(applyFog(col, in.wpos, U), in.wpos, fowTex, U), alpha);
}

// ---------------------------------------------------------------- objects
struct ObjOut {
    float4 pos [[position]];
    float3 wpos;
    float3 opos;      // object space: keeps panelling fixed to the hull
    float3 nrm;
    float4 colRough;
    float4 mte;       // metal, team weight, emissive, ao
    float4 fx;        // emissive gain, damage flash, dissolve, build cutoff Y
    float3 teamCol;
    float4 lightClip;
};

vertex ObjOut objectVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                       device const MVert* verts [[buffer(0)]],
                       constant FrameUniforms& U [[buffer(1)]],
                       device const Instance* insts [[buffer(2)]]) {
    MVert v = verts[vid];
    Instance I = insts[iid];
    float4 wp = I.model * float4(float3(v.pos), 1.0);
    float3 wn = normalize((I.model * float4(float3(v.nrm), 0.0)).xyz);
    ObjOut o;
    o.pos = U.viewProj * wp;
    o.wpos = wp.xyz;
    o.opos = float3(v.pos);
    o.nrm = wn;
    o.colRough = float4(v.colRough.rgb * I.tint.rgb, saturate(v.colRough.a + I.tint.a));
    o.mte = float4(saturate(v.mte.x + I.team.a), v.mte.y, v.mte.z, v.mte.w);
    o.fx = I.fx;
    o.teamCol = I.team.rgb;
    o.lightClip = U.lightViewProj * wp;
    return o;
}

fragment float4 objectFS(ObjOut in [[stage_in]],
                         constant FrameUniforms& U [[buffer(1)]],
                         depth2d<float> shadowMap [[texture(0)]],
                         texture2d_array<float> matAlbedo [[texture(1)]],
                         texture2d_array<float> matNormal [[texture(2)]],
                         texture2d<float> fowTex [[texture(4)]],
                         constant float4& objMat [[buffer(3)]],
                         sampler shadowSmp [[sampler(0)]],
                         sampler matSmp [[sampler(1)]]) {
    // Buildings extrude upward out of the ground while under construction.
    if (in.wpos.y > in.fx.w) discard_fragment();
    // Death dissolve: noise-thresholded erosion with a hot rim at the edge.
    float diss = in.fx.z;
    float dnoise = fbm2(in.wpos.xz * 3.1 + in.wpos.y * 1.7, 3);
    if (diss < 0.999 && dnoise > diss) discard_fragment();
    float dissEdge = (diss < 0.999) ? saturate(1.0 - (diss - dnoise) * 9.0) : 0.0;

    float3 N = normalize(in.nrm);
    float3 albedo = mix(in.colRough.rgb, in.teamCol, in.mte.y);
    float rough = in.colRough.a;
    float metallic = in.mte.x;

    // Armour panelling. These meshes are built from boxes and cylinders and carry
    // no UVs, so project on the dominant object-space axis -- for an axis-aligned
    // box that is exactly what a proper unwrap would give, at one sample instead
    // of triplanar's three. Emissive panels are left alone so they stay clean.
    // objMat: x = array slice, y = object-space uv scale, z = detail blend,
    // w = normal strength. Set once per mesh, so crystals get faceted rock and
    // machines get panel plating without a per-vertex material channel.
    if (U.misc.z > 0.001 && in.mte.z < 0.5) {
        float3 an = abs(normalize(in.nrm));
        // Object space, not world: otherwise the plating swims across the hull
        // as a unit turns.
        float2 pUV = (an.y > an.x && an.y > an.z) ? in.opos.xz
                   : ((an.x > an.z) ? in.opos.zy : in.opos.xy);
        pUV *= objMat.y;
        int slice = int(objMat.x);
        float4 tA = matAlbedo.sample(matSmp, pUV, slice);
        float4 tN = matNormal.sample(matSmp, pUV, slice);
        float3 detail = tA.rgb * 2.0;
        albedo *= mix(float3(1.0), detail, objMat.z);
        rough = saturate(rough + (0.5 - tA.r) * 0.30);
        N = applyDetailNormal(N, tN.rg * 2.0 - 1.0, objMat.w);
    }

    float3 V = normalize(U.camPos.xyz - in.wpos);
    float3 L = normalize(U.sunDir.xyz);
    float NdL = dot(N, L);
    float shadow = sampleShadow(shadowMap, shadowSmp, in.lightClip, NdL, U.misc.y);

    float3 col = pbrDirect(N, V, L, albedo, rough, metallic, U.sunColor.rgb * U.sunDir.w) * shadow;
    col += ambientIBL(N, albedo, in.mte.w, U.sunColor.w, U.fog.rgb);

    // Emissive panels, boosted by the per-instance gain.
    float3 emis = albedo * in.mte.z * (1.0 + in.fx.x);
    col += emis;

    // Fresnel rim tinted by faction colour: silhouettes stay readable at RTS zoom.
    float rim = pow(1.0 - saturate(dot(N, V)), 3.5);
    col += in.teamCol * rim * 0.35;

    // Damage flash: modulate the shaded colour rather than replacing it, so a
    // unit under sustained fire still reads as a shape instead of a white blob.
    float flash = saturate(in.fx.y);
    col = mix(col, col * float3(2.4, 0.95, 0.55), flash * 0.55);
    col += float3(0.85, 0.22, 0.08) * flash * 0.20;
    col += float3(1.0, 0.45, 0.12) * dissEdge * 3.5;

    // The construction scan line glows just under the cutoff plane.
    float band = saturate(1.0 - (in.fx.w - in.wpos.y) * 3.0);
    col += float3(0.35, 0.85, 1.0) * band * band * 1.6 * step(in.fx.w, 900.0);

    return float4(applyFOW(applyFog(col, in.wpos, U), in.wpos, fowTex, U), 1.0);
}

// ---------------------------------------------------------------- shadow pass
vertex float4 terrainShadowVS(uint vid [[vertex_id]],
                              device const TVert* verts [[buffer(0)]],
                              constant FrameUniforms& U [[buffer(1)]]) {
    return U.lightViewProj * float4(float3(verts[vid].pos), 1.0);
}

vertex float4 objectShadowVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                             device const MVert* verts [[buffer(0)]],
                             constant FrameUniforms& U [[buffer(1)]],
                             device const Instance* insts [[buffer(2)]]) {
    return U.lightViewProj * (insts[iid].model * float4(float3(verts[vid].pos), 1.0));
}
)MSL";

} // namespace sf

namespace sf {

inline const char* kShaderSource3 = R"MSL(
// ---------------------------------------------------------------- billboards
struct BBOut {
    float4 pos [[position]];
    float3 wpos;     // for the fog-of-war lookup
    float2 uv;
    float4 color;
    float  kind;
    float  phase;    // 0..1 through the sprite sheet
    float  param;    // per-kind extra; scorch uses it to pick a mark
};

vertex BBOut billboardVS(uint vid [[vertex_id]], uint iid [[instance_id]],
                         device const BBoard* bbs [[buffer(0)]],
                         constant FrameUniforms& U [[buffer(1)]]) {
    BBoard b = bbs[iid];
    const float2 corners[6] = { float2(-1,-1), float2(1,-1), float2(1,1),
                                float2(-1,-1), float2(1,1), float2(-1,1) };
    float2 c = corners[vid];

    float3 right, up;
    // Selection rings, shockwaves and scorch marks lie flat on the ground;
    // everything else -- puffs, sparks, fireballs, smoke -- faces the camera.
    bool flat = (b.kind > 1.5 && b.kind < 3.5) || (b.kind > 4.5 && b.kind < 5.5);
    if (flat) {
        right = float3(1, 0, 0); up = float3(0, 0, 1);
    } else {
        // Camera-facing: the view matrix rows are the camera basis in world space.
        right = float3(U.view[0][0], U.view[1][0], U.view[2][0]);
        up    = float3(U.view[0][1], U.view[1][1], U.view[2][1]);
    }
    float s = sin(b.rot), co = cos(b.rot);
    float2 r = float2(c.x * co - c.y * s, c.x * s + c.y * co);
    // Fireballs and smoke plumes grow upward out of their origin instead of
    // being centred on it. A camera-facing quad centred on a blast sinks half
    // its height into the terrain, and the depth test then slices the bottom
    // off along a dead-straight horizontal line.
    float lift = (!flat && b.kind > 3.5) ? b.size * 0.85 : 0.0;
    float3 wp = float3(b.pos) + float3(0.0, lift, 0.0) + (right * r.x + up * r.y) * b.size;

    BBOut o;
    o.pos = U.viewProj * float4(wp, 1.0);
    o.wpos = wp;
    o.uv = c;
    o.color = b.color;
    o.kind = b.kind;
    o.phase = b.fade;
    o.param = b.param;
    return o;
}

// Cell assignments for the 4x4 particle sheet, read off the artwork. Ten of the
// sixteen cells are smoke and dust, six are sparks and flashes; drawing from the
// right group is what keeps a muzzle flash from rendering as a smoke curl.
constant int kPuffCells[10]  = { 0, 1, 3, 5, 7, 8, 9, 12, 13, 15 };
constant int kSparkCells[6]  = { 2, 4, 6, 10, 11, 14 };

fragment float4 billboardFS(BBOut in [[stage_in]],
                            constant FrameUniforms& U [[buffer(1)]],
                            texture2d<float> spriteTex [[texture(3)]],
                            texture2d<float> fowTex [[texture(4)]],
                            texture2d<float> particleTex [[texture(9)]],
                            sampler matSmp [[sampler(1)]]) {
    // Effects are live information, so unlike terrain they are cut entirely by
    // the fog rather than dimmed to a remembered version.
    float fow = (U.texParams.y < 0.001) ? 1.0
              : mix(1.0, fowAt(fowTex, in.wpos, U).r, U.texParams.y);
    // Kind 4 plays the 4x4 explosion sheet. It renders additively against a black
    // background, so the sheet needs no alpha channel -- which is why a JPEG works.
    if (in.kind > 3.5) {
        float2 uv01 = in.uv * 0.5 + 0.5;
        float f = clamp(in.phase, 0.0, 0.9999) * 16.0;
        float idx = floor(f);
        float2 cell = float2(fmod(idx, 4.0), floor(idx * 0.25));
        float2 suv = (cell + clamp(uv01, 0.001, 0.999)) * 0.25;
        float3 c = spriteTex.sample(matSmp, suv).rgb;
        return float4(c * in.color.rgb * in.color.a * fow, 0.0);
    }
    float r = length(in.uv);
    float a;
    // Kinds 0 and 1 are the workhorses -- every muzzle flash, impact, dust puff,
    // engine trail and debris spark in the game -- and analytically they were a
    // single perfect circle each. `param` picks a cell so a burst is sixteen
    // different shapes rather than sixteen copies of one.
    if (in.kind < 1.5 && U.texFlags.y > 0.5) {
        float sel = clamp(in.param, 0.0, 0.9999);
        int idx = (in.kind < 0.5) ? kPuffCells[int(sel * 10.0)]
                                  : kSparkCells[int(sel * 6.0)];
        float2 cell = float2(float(idx & 3), float(idx >> 2));
        float2 uv01 = clamp(in.uv * 0.5 + 0.5, 0.004, 0.996);
        a = particleTex.sample(matSmp, (cell + uv01) * 0.25).r;
        // The painted shapes carry less coverage than a solid analytic falloff,
        // so they need a gain to land at the brightness the effects were tuned
        // for. Sparks are drawn hotter than smoke, as before.
        a *= (in.kind < 0.5) ? 1.7 : 2.3;
    } else if (in.kind < 0.5) {
        // Soft volumetric puff.
        a = pow(saturate(1.0 - r), 2.2);
    } else if (in.kind < 1.5) {
        // Hot spark: tight core with a bright centre.
        a = pow(saturate(1.0 - r), 5.0) + pow(saturate(1.0 - r * 2.4), 12.0) * 1.5;
    } else if (in.kind < 2.5) {
        // Selection / status ring.
        a = saturate(1.0 - abs(r - 0.86) * 13.0) * step(r, 1.0);
    } else {
        // Expanding shockwave: thin leading edge with an inner falloff.
        a = saturate(1.0 - abs(r - 0.92) * 9.0) * step(r, 1.0);
        a += saturate(1.0 - r) * 0.16;
    }
    a *= in.color.a * fow;
    return float4(in.color.rgb * a, a);
}

// Alpha-blended sprites: ground scorch and smoke. These share billboardVS with
// the additive particles but need their own pass -- additive blending can only
// brighten, and a burn mark has to darken the ground underneath it.
fragment float4 decalFS(BBOut in [[stage_in]],
                        constant FrameUniforms& U [[buffer(1)]],
                        texture2d<float> fowTex [[texture(4)]],
                        texture2d<float> scorchTex [[texture(5)]],
                        texture2d<float> smokeTex [[texture(6)]],
                        sampler matSmp [[sampler(1)]]) {
    float2 uv01 = in.uv * 0.5 + 0.5;
    float a;
    if (in.kind < 5.5) {
        // Scorch: a 2x2 sheet of burn marks, authored bright-on-black so the
        // coverage survives JPEG. `param` picks which of the four.
        int idx = int(clamp(in.param, 0.0, 0.999) * 4.0);
        float2 cell = float2(float(idx & 1), float(idx >> 1));
        a = scorchTex.sample(matSmp, (cell + clamp(uv01, 0.004, 0.996)) * 0.5).r;
        // Only a small floor: squaring the mask looked tidier on the two bright
        // marks but erased the diffuse soot smudge almost entirely.
        a = saturate((a - 0.05) * 1.06);
    } else {
        // Smoke: a 16-frame plume. Unlike the explosion sheet this one is not
        // clean black between the plumes -- the generator left a haze averaging
        // 0.11 in the cell corners, which as coverage would draw the whole quad
        // as a visible grey rectangle. Hence the floor subtraction, and a border
        // mask because the later frames run to the edge of their cell.
        float f = floor(clamp(in.phase, 0.0, 0.9999) * 16.0);
        float2 cell = float2(fmod(f, 4.0), floor(f * 0.25));
        a = smokeTex.sample(matSmp, (cell + clamp(uv01, 0.004, 0.996)) * 0.25).r;
        a = saturate((a - 0.16) * 1.19);
        float2 d = abs(in.uv);
        a *= 1.0 - smoothstep(0.50, 0.98, max(d.x, d.y));
    }
    a *= in.color.a;
    if (U.texParams.y > 0.001) a *= mix(1.0, fowAt(fowTex, in.wpos, U).r, U.texParams.y);
    return float4(in.color.rgb, a);
}

// ---------------------------------------------------------------- post chain
struct FSOut { float4 pos [[position]]; float2 uv; };

vertex FSOut fullscreenVS(uint vid [[vertex_id]]) {
    float2 p = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    FSOut o;
    o.pos = float4(p, 0.0, 1.0);
    o.uv = float2(p.x * 0.5 + 0.5, -p.y * 0.5 + 0.5);
    return o;
}

constexpr sampler linClamp(filter::linear, mip_filter::none, address::clamp_to_edge);

// Soft-knee threshold, then a 4-tap box downsample in one pass.
fragment float4 brightPassFS(FSOut in [[stage_in]],
                             texture2d<float> src [[texture(0)]],
                             constant float4& p [[buffer(0)]]) {
    float2 t = p.zw;   // source texel size
    float3 c = (src.sample(linClamp, in.uv + float2(-t.x, -t.y)).rgb +
                src.sample(linClamp, in.uv + float2( t.x, -t.y)).rgb +
                src.sample(linClamp, in.uv + float2(-t.x,  t.y)).rgb +
                src.sample(linClamp, in.uv + float2( t.x,  t.y)).rgb) * 0.25;
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    float knee = max(lum - p.x, 0.0);
    float w = knee / max(lum, 1e-4);
    return float4(c * w * p.y, 1.0);
}

fragment float4 downsampleFS(FSOut in [[stage_in]],
                             texture2d<float> src [[texture(0)]],
                             constant float4& p [[buffer(0)]]) {
    float2 t = p.zw;
    float3 c = (src.sample(linClamp, in.uv + float2(-t.x, -t.y)).rgb +
                src.sample(linClamp, in.uv + float2( t.x, -t.y)).rgb +
                src.sample(linClamp, in.uv + float2(-t.x,  t.y)).rgb +
                src.sample(linClamp, in.uv + float2( t.x,  t.y)).rgb) * 0.25;
    return float4(c, 1.0);
}

// Separable 9-tap Gaussian; `dir` carries the axis scaled by texel size.
fragment float4 blurFS(FSOut in [[stage_in]],
                       texture2d<float> src [[texture(0)]],
                       constant float4& p [[buffer(0)]]) {
    float2 d = p.xy;
    const float w[5] = { 0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162 };
    float3 c = src.sample(linClamp, in.uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        c += src.sample(linClamp, in.uv + d * float(i)).rgb * w[i];
        c += src.sample(linClamp, in.uv - d * float(i)).rgb * w[i];
    }
    return float4(c, 1.0);
}

// Narkowicz ACES approximation.
static inline float3 acesTonemap(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

fragment float4 compositeFS(FSOut in [[stage_in]],
                            texture2d<float> hdr   [[texture(0)]],
                            texture2d<float> bloom [[texture(1)]],
                            texture2d<float> bloom2[[texture(2)]],
                            constant float4& p [[buffer(0)]]) {
    float2 uv = in.uv;
    float2 fromCentre = uv - 0.5;
    float r2 = dot(fromCentre, fromCentre);

    // Very slight lateral chromatic aberration toward the frame edges.
    float ca = p.z * r2;
    float3 c;
    c.r = hdr.sample(linClamp, uv + fromCentre * ca).r;
    c.g = hdr.sample(linClamp, uv).g;
    c.b = hdr.sample(linClamp, uv - fromCentre * ca).b;

    c += bloom.sample(linClamp, uv).rgb  * p.y;
    c += bloom2.sample(linClamp, uv).rgb * p.y * 0.85;

    c *= p.x;                          // exposure
    c = acesTonemap(c);
    c = pow(c, float3(1.0 / 2.2));     // to display gamma

    // Slight saturation lift and a soft vignette.
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = mix(float3(lum), c, 1.20);
    c *= 1.0 - saturate(r2 * 0.95) * 0.20;

    // Fine film grain keeps gradients from banding on flat sky.
    float g = hash21(uv * 1024.0 + p.w) - 0.5;
    c += g * 0.016;

    return float4(saturate(c), 1.0);
}

// ---------------------------------------------------------------- HUD
struct UIOut {
    float4 pos [[position]];
    float2 uv;
    float4 col;
};

vertex UIOut uiVS(uint vid [[vertex_id]],
                  device const UIVert* verts [[buffer(0)]],
                  constant float4& screen [[buffer(1)]]) {
    UIVert v = verts[vid];
    UIOut o;
    // Pixel coordinates, origin top-left, into clip space.
    o.pos = float4(v.xy.x * screen.x * 2.0 - 1.0, 1.0 - v.xy.y * screen.y * 2.0, 0.0, 1.0);
    o.uv = v.uv;
    o.col = v.rgba;
    return o;
}

fragment float4 uiFS(UIOut in [[stage_in]], texture2d<float> atlas [[texture(0)]]) {
    // One pipeline serves both the glyph atlas (white RGB, coverage in alpha)
    // and the minimap texture (opaque RGB), so both modulate the same way.
    float4 t = atlas.sample(linClamp, in.uv);
    return float4(in.col.rgb * t.rgb, in.col.a * t.a);
}
)MSL";

} // namespace sf
