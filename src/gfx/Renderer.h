// Renderer.h — pure C++ facade over the Metal backend. No Objective-C leaks out,
// so the simulation and game layers stay portable C++.
#pragma once
#include <vector>
#include <string>
#include "RenderTypes.h"
#include "MeshGen.h"

namespace sf {

class Terrain;

// Everything the renderer needs for one frame. The game layer fills this in and
// hands it over; the renderer never reaches back into game state.
struct RenderFrame {
    m4 viewProj, view, invViewProj;
    v3 camPos{};        float time = 0.0f;
    v3 sunDir{};        float sunIntensity = 3.2f;
    v3 sunColor{};      float ambient = 1.0f;
    v3 fogColor{};      float fogDensity = 0.0016f;
    v3 shadowFocus{};   float waterLevel = 0.0f;
    float exposure = 1.05f, bloomIntensity = 0.55f, bloomThreshold = 1.15f;

    std::vector<InstanceData> instances[MESH_COUNT];
    std::vector<Billboard>    billboards;   // additive: fire, sparks, rings
    // Alpha-blended sprites: ground scorch first, then smoke. Additive blending
    // can only ever brighten, so a burn mark has to live in its own pass.
    std::vector<Billboard>    decals;
    std::vector<UIVertex>     uiVerts;       // drawn against the glyph atlas
    std::vector<UIVertex>     minimapVerts;  // drawn against the minimap texture

    // Player-side fog of war, as an fowN x fowN grid of (visible, explored)
    // byte pairs covering the whole map. Empty or fowStrength 0 disables it.
    std::vector<uint8_t> fow;
    int   fowN = 0;
    float fowStrength = 0.0f;
    float mapSize = 1.0f;

    void clear() {
        for (auto& v : instances) v.clear();
        billboards.clear(); decals.clear(); uiVerts.clear(); minimapVerts.clear();
    }
};

struct RenderStats {
    int   drawCalls = 0;
    int   instances = 0;
    int   triangles = 0;
    double gpuMs = 0.0;          // whole command buffer; includes the present wait
    // Real per-pass GPU time from timestamp counters. Unlike gpuMs these are
    // pure execution and are what the optimisation work is driven by.
    // NOTE: these per-pass spans overlap. A tile-based GPU runs the vertex
    // stage of the next pass during the fragment stage of the current one, so
    // they must not be summed. gpuFrameMs is the real figure.
    double shadowMs = 0.0, sceneMs = 0.0, bloomMs = 0.0, compositeMs = 0.0;
    double gpuFrameMs = 0.0;   // first pass start -> last pass end
    double texMB = 0.0;
};

// Quality knobs. Sized for the Apple A18 Pro's 5-core GPU: at 2560x1526 the
// frame is fragment- and bandwidth-bound, so these are the levers that matter.
struct RenderSettings {
    float renderScale = 1.0f;   // drawable size multiplier; the compositor upscales
    // 2x rather than 4x: measured on the A18 Pro's 5-core GPU this saves 2.9 ms
    // of a 13.2 ms frame for very little visible difference. Note 1x is *slower*
    // than 2x here -- turning MSAA off abandons the memoryless tile-memory path
    // and writes the HDR target through main memory instead.
    int   msaa        = 2;
    int   shadowRes   = 2048;
    bool  bloom       = true;
};

class Renderer {
public:
    Renderer();
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // `caMetalLayer` is a CAMetalLayer*. Returns false and fills `err` on failure.
    bool init(void* caMetalLayer, const Terrain& terrain, std::string& err,
              const RenderSettings& rs = RenderSettings{});
    // Changing MSAA or shadow resolution rebuilds pipelines/targets.
    void applySettings(const RenderSettings& rs);
    RenderSettings settings() const;
    void resize(int pixelW, int pixelH);
    // Re-uploads terrain, water and minimap resources for a freshly generated map.
    void rebuildTerrain(const Terrain& terrain);
    void render(const RenderFrame& frame);
    // Grabs the next presented frame to a PNG (used for automated verification).
    void requestCapture(const std::string& path);
    bool captureComplete() const;

    // --- HUD construction helpers (glyph metrics live beside the atlas) ------
    float textWidth(const char* s, float size) const;
    float lineHeight(float size) const;
    void  text(std::vector<UIVertex>& out, const char* s, float x, float y,
               float size, v4 color) const;
    void  rect(std::vector<UIVertex>& out, float x, float y, float w, float h, v4 color) const;
    // Outline drawn as four rectangles; used for the selection marquee and panels.
    void  frame(std::vector<UIVertex>& out, float x, float y, float w, float h,
                float t, v4 color) const;
    // Emits a quad sampling the whole minimap texture; goes in `minimapVerts`.
    void  minimapQuad(std::vector<UIVertex>& out, float x, float y, float w, float h,
                      v4 tint) const;

    const MeshRange& meshRange(int id) const;
    RenderStats stats() const;

    struct Impl;   // Metal state; defined in Renderer.mm
private:
    Impl* p_ = nullptr;
};

} // namespace sf
