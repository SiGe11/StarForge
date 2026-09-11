// Renderer.mm — Metal backend. Render graph per frame:
//   1. shadow    : depth-only cascade from the sun
//   2. main      : sky -> terrain -> instanced objects -> water -> particles
//                  (4x MSAA into a memoryless HDR target, resolved on store)
//   3. bloom     : threshold -> blur(half) -> downsample -> blur(quarter)
//   4. composite : tonemap + HUD, straight to the drawable
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <CoreText/CoreText.h>
#import <ImageIO/ImageIO.h>
#include <cstring>
#include <unistd.h>
#import <mach/mach_time.h>
#include "Renderer.h"
#include "Shaders.h"
#include "../sim/Terrain.h"
#include <string>

namespace sf {

// Half the bytes per pixel of RGBA16Float across every HDR target and the whole
// bloom chain. Nothing here reads destination alpha, and 11-bit float mantissas
// are plenty for pre-tonemap colour.
static constexpr MTLPixelFormat kHDRFormat = MTLPixelFormatRG11B10Float;
static constexpr int   kFramesInFlight = 3;
static constexpr int   kMaxInstances  = 8192;
static constexpr int   kMaxBillboards = 16384;
static constexpr int   kMaxUIVerts    = 98304;
static constexpr float kFontSize      = 44.0f;
static constexpr int   kGlyphCols     = 16;
static constexpr int   kGlyphRows     = 6;
static constexpr int   kWhiteGlyph    = 95;   // last cell is a solid white block

struct Glyph { float u0, v0, u1, v1, advance; };

struct Renderer::Impl {
    id<MTLDevice>       dev = nil;
    id<MTLCommandQueue> queue = nil;
    CAMetalLayer*       layer = nil;

    id<MTLRenderPipelineState> psSky, psTerrain, psWater, psObject, psBillboard, psDecal;
    id<MTLRenderPipelineState> psTerrainShadow, psObjectShadow;
    id<MTLRenderPipelineState> psBright, psDown, psBlur, psComposite, psUI;
    id<MTLDepthStencilState>   dsOpaque, dsTestNoWrite, dsAlways, dsShadow;
    id<MTLSamplerState>        shadowSampler;

    id<MTLBuffer> terrainVB, terrainIB, waterVB, waterIB, meshVB, meshIB;
    uint32_t terrainIdxCount = 0, waterIdxCount = 0;
    MeshRange ranges[MESH_COUNT];

    id<MTLBuffer> uniformBuf[kFramesInFlight];
    id<MTLBuffer> instanceBuf[kFramesInFlight];
    id<MTLBuffer> billboardBuf[kFramesInFlight];
    id<MTLBuffer> decalBuf[kFramesInFlight];
    id<MTLBuffer> uiBuf[kFramesInFlight];

    id<MTLTexture> shadowMap, hdrMSAA, depthMSAA, hdrResolve;
    id<MTLTexture> bloomHalfA, bloomHalfB, bloomQtrA, bloomQtrB;
    id<MTLTexture> fontTex, minimapTex;
    // Material textures. Two 4-slice arrays (ground / cliff / armour / crystal)
    // plus the sprite sheets and masks. Everything is mipmapped: on a 60 GB/s
    // part the bandwidth saved by mips matters more than the memory they cost.
    id<MTLTexture> matAlbedo, matNormal, explosionTex;
    id<MTLTexture> waterNrm;                 // ripple height map, Sobel'd to a normal
    id<MTLTexture> scorchTex, smokeTex;      // single-channel coverage masks
    // Three more single-channel maps, each replacing a block of per-pixel noise.
    // Clouds is the big one: the sky pass runs full-screen with the depth test
    // off, and waterFS calls skyColor() again for reflections, so the fBm it
    // replaces was being evaluated twice over.
    id<MTLTexture> cloudTex, particleTex, macroTex;
    float waterTexStrength = 0.0f;
    // Player fog of war: RG8, one texel per simulation visibility cell, refilled
    // every frame. Tiny enough that a shared-storage upload is free.
    id<MTLTexture> fowTex;
    int fowN = 0;
    id<MTLSamplerState> repeatSampler;
    float texStrength = 0.0f;      // 0 when assets are missing: pure procedural
    size_t texBytes = 0;

    // One counter buffer per frame in flight: a single shared buffer gets
    // overwritten by later frames before the completion handler resolves it,
    // which mixes timestamps from different frames into nonsense deltas.
    id<MTLCounterSampleBuffer> gpuCounters[kFramesInFlight] = {nil, nil, nil};
    bool   countersOK = false;
    double gpuTickToMs = 0.0;
    dispatch_semaphore_t sem = nil;
    int frameIndex = 0;
    int width = 1, height = 1;

    id<MTLTexture> captureTex;
    std::string capturePath;
    bool captureDone = true;

    RenderSettings cfg;
    id<MTLLibrary> lib = nil;        // kept so pipelines can be rebuilt on MSAA change

    Glyph glyphs[96];
    float cellW = 1, cellH = 1, padX = 0;   // normalised by kFontSize
    RenderStats stats;
};

// ---------------------------------------------------------------- image loading
namespace {

struct RawImage { int w = 0, h = 0; std::vector<uint8_t> px; };   // RGBA8

// Decodes and downscales in one step. The source art is 1254px; a power-of-two
// 1024 mips cleanly and costs 40% less memory, which is the scarcer resource on
// an 8 GB unified-memory machine. The originals on disk are left untouched.
bool loadImageRGBA(const std::string& path, int maxSize, RawImage& out) {
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        NULL, (const UInt8*)path.c_str(), (CFIndex)path.size(), false);
    if (!url) return false;
    CGImageSourceRef src = CGImageSourceCreateWithURL(url, NULL);
    CFRelease(url);
    if (!src) return false;

    CFMutableDictionaryRef opts = CFDictionaryCreateMutable(
        NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(opts, kCGImageSourceCreateThumbnailFromImageAlways, kCFBooleanTrue);
    CFDictionarySetValue(opts, kCGImageSourceShouldCache, kCFBooleanFalse);
    CFNumberRef n = CFNumberCreate(NULL, kCFNumberIntType, &maxSize);
    CFDictionarySetValue(opts, kCGImageSourceThumbnailMaxPixelSize, n);
    CFRelease(n);
    CGImageRef img = CGImageSourceCreateThumbnailAtIndex(src, 0, opts);
    CFRelease(opts);
    CFRelease(src);
    if (!img) return false;

    out.w = (int)CGImageGetWidth(img);
    out.h = (int)CGImageGetHeight(img);
    out.px.assign((size_t)out.w * out.h * 4, 0);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(out.px.data(), out.w, out.h, 8, out.w * 4, cs,
                                             (CGBitmapInfo)kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    if (ctx) {
        CGContextDrawImage(ctx, CGRectMake(0, 0, out.w, out.h), img);
        CGContextRelease(ctx);
    }
    CGColorSpaceRelease(cs);
    CGImageRelease(img);
    return ctx != NULL;
}

// Rescale so the texture's mean *linear* luminance is 0.5. The shader then
// multiplies by two to get a detail term centred on 1.0, which modulates the
// procedural albedo without shifting overall exposure. Doing this once here
// keeps the shader free of per-texture magic numbers and means any texture
// dropped into assets/ is automatically exposure-matched.
void normaliseMeanLuminance(RawImage& img) {
    auto toLinear = [](float c) {
        return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    };
    auto toSRGB = [](float c) {
        return (c <= 0.0031308f) ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    };
    double sum = 0.0;
    const size_t n = img.px.size() / 4;
    for (size_t i = 0; i < n; i++) {
        const uint8_t* p = &img.px[i * 4];
        sum += 0.2126 * toLinear(p[0] / 255.0f) + 0.7152 * toLinear(p[1] / 255.0f) +
               0.0722 * toLinear(p[2] / 255.0f);
    }
    float mean = (float)(sum / std::max<size_t>(1, n));
    if (mean < 1e-4f) return;
    float scale = 0.5f / mean;
    for (size_t i = 0; i < n; i++) {
        uint8_t* p = &img.px[i * 4];
        for (int c = 0; c < 3; c++)
            p[c] = (uint8_t)(saturate(toSRGB(saturate(toLinear(p[c] / 255.0f) * scale))) * 255.0f);
    }
}

// Sobel over luminance -> tangent-space normal. Image generators cannot produce
// a valid normal map, but they do produce exactly what this needs: flat-lit art
// where recesses are darker. Sampling wraps, so tiling stays seamless.
void deriveNormalHeight(const RawImage& src, float strength, std::vector<uint8_t>& out) {
    const int w = src.w, h = src.h;
    out.assign((size_t)w * h * 4, 0);
    auto lum = [&](int x, int y) {
        x = (x % w + w) % w; y = (y % h + h) % h;
        const uint8_t* p = &src.px[((size_t)y * w + x) * 4];
        return (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * (1.0f / 255.0f);
    };
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float l = lum(x - 1, y), r = lum(x + 1, y);
            float d = lum(x, y - 1), u = lum(x, y + 1);
            v3 nrm = normalize(v3{(l - r) * strength, (d - u) * strength, 1.0f});
            uint8_t* o = &out[((size_t)y * w + x) * 4];
            o[0] = (uint8_t)(saturate(nrm.x * 0.5f + 0.5f) * 255.0f);
            o[1] = (uint8_t)(saturate(nrm.y * 0.5f + 0.5f) * 255.0f);
            o[2] = (uint8_t)(saturate(lum(x, y)) * 255.0f);   // height, for parallax-ish use
            o[3] = 255;
        }
}

// Collapse an RGBA image to a one-byte coverage mask. The source art is
// bright-on-black, so luminance *is* the coverage. Deliberately left in sRGB
// rather than linearised: for a hand-painted mask the perceptual ramp is the
// one that looks right, and it is also what survives JPEG best.
void toR8(const RawImage& src, std::vector<uint8_t>& out) {
    out.resize((size_t)src.w * src.h);
    for (size_t i = 0; i < out.size(); i++) {
        const uint8_t* px = &src.px[i * 4];
        out[i] = (uint8_t)((px[0] * 77 + px[1] * 150 + px[2] * 29) >> 8);
    }
}

// Assets sit next to the binary or next to the working directory.
std::string findAsset(const char* name) {
    std::string local = std::string("assets/") + name;
    if (access(local.c_str(), R_OK) == 0) return local;
    @autoreleasepool {
        NSString* exe = [[NSBundle mainBundle] executablePath];
        if (exe) {
            NSString* dir = [exe stringByDeletingLastPathComponent];
            NSString* p = [NSString stringWithFormat:@"%@/assets/%s", dir, name];
            if (access([p UTF8String], R_OK) == 0) return std::string([p UTF8String]);
        }
    }
    return std::string();
}

} // namespace

// ---------------------------------------------------------------- font atlas
static void buildFontAtlas(Renderer::Impl* p) {
    CTFontRef font = CTFontCreateWithName(CFSTR("Menlo-Bold"), kFontSize, NULL);
    if (!font) font = CTFontCreateUIFontForLanguage(kCTFontUIFontUser, kFontSize, NULL);

    UniChar chars[96];
    CGGlyph glyphIds[96];
    CGSize  advances[96];
    for (int i = 0; i < 96; i++) chars[i] = (UniChar)(32 + i);
    CTFontGetGlyphsForCharacters(font, chars, glyphIds, 96);
    CTFontGetAdvancesForGlyphs(font, kCTFontOrientationHorizontal, glyphIds, advances, 96);

    float maxAdv = 0;
    for (int i = 0; i < 96; i++) maxAdv = std::max(maxAdv, (float)advances[i].width);
    const float padX = 3.0f;
    const int cellW = (int)std::ceil(maxAdv + padX * 2.0f);
    const int cellH = (int)std::ceil(kFontSize * 1.45f);
    const int aw = cellW * kGlyphCols, ah = cellH * kGlyphRows;
    const float ascent = (float)CTFontGetAscent(font);
    const float blFromTop = std::floor(kFontSize * 0.14f) + ascent;

    std::vector<uint8_t> cov((size_t)aw * ah, 0);
    CGContextRef ctx = CGBitmapContextCreate(cov.data(), aw, ah, 8, aw, NULL, kCGImageAlphaOnly);
    CGContextSetGrayFillColor(ctx, 1.0, 1.0);
    CGContextSetShouldAntialias(ctx, true);
    CGContextSetShouldSmoothFonts(ctx, false);

    for (int i = 0; i < 96; i++) {
        int cx = i % kGlyphCols, cy = i / kGlyphCols;
        float left = cx * (float)cellW, top = cy * (float)cellH;
        if (i != kWhiteGlyph) {
            // CoreGraphics origin is bottom-left; convert from our top-left cells.
            CGPoint pt = CGPointMake(left + padX, ah - (top + blFromTop));
            CTFontDrawGlyphs(font, &glyphIds[i], &pt, 1, ctx);
        }
        p->glyphs[i] = Glyph{ left / aw, top / ah,
                              (left + cellW) / aw, (top + cellH) / ah,
                              (float)advances[i].width / kFontSize };
    }
    CGContextRelease(ctx);
    CFRelease(font);

    // Fill the reserved cell so solid HUD rectangles can share this pipeline.
    {
        int cx = kWhiteGlyph % kGlyphCols, cy = kWhiteGlyph / kGlyphCols;
        for (int y = cy * cellH; y < (cy + 1) * cellH; y++)
            for (int x = cx * cellW; x < (cx + 1) * cellW; x++)
                cov[(size_t)y * aw + x] = 255;
        // Sample the middle of the cell to stay clear of bilinear bleed.
        float u = (cx + 0.5f) * cellW / aw, v = (cy + 0.5f) * cellH / ah;
        p->glyphs[kWhiteGlyph] = Glyph{u, v, u, v, 0.0f};
    }

    std::vector<uint8_t> rgba((size_t)aw * ah * 4);
    for (size_t i = 0; i < (size_t)aw * ah; i++) {
        rgba[i * 4 + 0] = 255; rgba[i * 4 + 1] = 255;
        rgba[i * 4 + 2] = 255; rgba[i * 4 + 3] = cov[i];
    }

    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:aw height:ah mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    p->fontTex = [p->dev newTextureWithDescriptor:td];
    [p->fontTex replaceRegion:MTLRegionMake2D(0, 0, aw, ah)
                  mipmapLevel:0 withBytes:rgba.data() bytesPerRow:aw * 4];

    p->cellW = cellW / kFontSize;
    p->cellH = cellH / kFontSize;
    p->padX  = padX / kFontSize;
}

// ---------------------------------------------------------------- pipelines
static id<MTLRenderPipelineState> makePipe(id<MTLDevice> dev, id<MTLLibrary> lib,
                                           NSString* vs, NSString* fs,
                                           MTLPixelFormat color, MTLPixelFormat depth,
                                           NSUInteger samples, int blendMode,
                                           NSString* label, std::string& err) {
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.label = label;
    d.vertexFunction = [lib newFunctionWithName:vs];
    d.fragmentFunction = fs ? [lib newFunctionWithName:fs] : nil;
    d.rasterSampleCount = samples;
    if (color != MTLPixelFormatInvalid) {
        d.colorAttachments[0].pixelFormat = color;
        if (blendMode) {
            d.colorAttachments[0].blendingEnabled = YES;
            d.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
            d.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
            if (blendMode == 1) {  // straight alpha
                d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
                d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
                d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            } else {               // premultiplied additive (particles)
                d.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
                d.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
                d.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
                d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
            }
        }
    }
    d.depthAttachmentPixelFormat = depth;
    NSError* e = nil;
    id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:d error:&e];
    if (!ps) err += std::string([[e localizedDescription] UTF8String]) + "\n";
    return ps;
}

// Uploads every resource derived from the height field. Called on init and
// again whenever a new map is generated for a fresh match.
static void uploadTerrain(Renderer::Impl* p, const Terrain& terrain) {
    auto mkBuf = [&](const void* data, size_t bytes, NSString* label) -> id<MTLBuffer> {
        if (!bytes) return nil;
        id<MTLBuffer> b = [p->dev newBufferWithBytes:data length:bytes
                                             options:MTLResourceStorageModeShared];
        b.label = label;
        return b;
    };
    const auto& tv = terrain.vertices();
    const auto& ti = terrain.indices();
    p->terrainVB = mkBuf(tv.data(), tv.size() * sizeof(TerrainVertex), @"terrainVB");
    p->terrainIB = mkBuf(ti.data(), ti.size() * sizeof(uint32_t), @"terrainIB");
    p->terrainIdxCount = (uint32_t)ti.size();

    std::vector<TerrainVertex> wv; std::vector<uint32_t> wi;
    terrain.buildWaterMesh(wv, wi);
    p->waterIdxCount = (uint32_t)wi.size();
    p->waterVB = mkBuf(wv.data(), wv.size() * sizeof(TerrainVertex), @"waterVB");
    p->waterIB = mkBuf(wi.data(), wi.size() * sizeof(uint32_t), @"waterIB");

    std::vector<uint8_t> mm;
    terrain.buildMinimapRGBA(mm);
    if (!p->minimapTex) {
        MTLTextureDescriptor* mmd =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:Terrain::N height:Terrain::N mipmapped:NO];
        mmd.usage = MTLTextureUsageShaderRead;
        mmd.storageMode = MTLStorageModeShared;
        p->minimapTex = [p->dev newTextureWithDescriptor:mmd];
    }
    [p->minimapTex replaceRegion:MTLRegionMake2D(0, 0, Terrain::N, Terrain::N)
                     mipmapLevel:0 withBytes:mm.data() bytesPerRow:Terrain::N * 4];
}

Renderer::Renderer() : p_(new Impl()) {}
Renderer::~Renderer() { delete p_; }

bool Renderer::init(void* caMetalLayer, const Terrain& terrain, std::string& err,
                    const RenderSettings& rs) {
    Impl* p = p_;
    p->cfg = rs;
    p->layer = (__bridge CAMetalLayer*)caMetalLayer;
    p->dev = MTLCreateSystemDefaultDevice();
    if (!p->dev) { err = "no Metal device"; return false; }
    p->queue = [p->dev newCommandQueue];
    p->layer.device = p->dev;
    p->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    p->layer.framebufferOnly = NO;   // permits --shot readback of the drawable
    p->sem = dispatch_semaphore_create(kFramesInFlight);

    // --- shaders --------------------------------------------------------
    std::string src = std::string(kShaderSource) + kShaderSource2 + kShaderSource3;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.mathMode = MTLMathModeFast;
    NSError* e = nil;
    id<MTLLibrary> lib = [p->dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                              options:opts error:&e];
    p->lib = lib;
    if (!lib) { err = std::string("shader compile failed:\n") + [[e localizedDescription] UTF8String]; return false; }

    const MTLPixelFormat HDR = kHDRFormat;
    const MTLPixelFormat DEP = MTLPixelFormatDepth32Float;
    const MTLPixelFormat BGRA = MTLPixelFormatBGRA8Unorm;
    p->psSky      = makePipe(p->dev, lib, @"skyVS", @"skyFS", HDR, DEP, 4, 0, @"sky", err);
    p->psTerrain  = makePipe(p->dev, lib, @"terrainVS", @"terrainFS", HDR, DEP, (NSUInteger)p->cfg.msaa, 0, @"terrain", err);
    p->psObject   = makePipe(p->dev, lib, @"objectVS", @"objectFS", HDR, DEP, (NSUInteger)p->cfg.msaa, 0, @"object", err);
    p->psWater    = makePipe(p->dev, lib, @"waterVS", @"waterFS", HDR, DEP, (NSUInteger)p->cfg.msaa, 1, @"water", err);
    p->psBillboard= makePipe(p->dev, lib, @"billboardVS", @"billboardFS", HDR, DEP, (NSUInteger)p->cfg.msaa, 2, @"particles", err);
    p->psDecal    = makePipe(p->dev, lib, @"billboardVS", @"decalFS",     HDR, DEP, (NSUInteger)p->cfg.msaa, 1, @"decals", err);
    p->psTerrainShadow = makePipe(p->dev, lib, @"terrainShadowVS", nil, MTLPixelFormatInvalid, DEP, 1, 0, @"terrainShadow", err);
    p->psObjectShadow  = makePipe(p->dev, lib, @"objectShadowVS", nil, MTLPixelFormatInvalid, DEP, 1, 0, @"objectShadow", err);
    p->psBright   = makePipe(p->dev, lib, @"fullscreenVS", @"brightPassFS", HDR, MTLPixelFormatInvalid, 1, 0, @"bright", err);
    p->psDown     = makePipe(p->dev, lib, @"fullscreenVS", @"downsampleFS", HDR, MTLPixelFormatInvalid, 1, 0, @"downsample", err);
    p->psBlur     = makePipe(p->dev, lib, @"fullscreenVS", @"blurFS", HDR, MTLPixelFormatInvalid, 1, 0, @"blur", err);
    p->psComposite= makePipe(p->dev, lib, @"fullscreenVS", @"compositeFS", BGRA, MTLPixelFormatInvalid, 1, 0, @"composite", err);
    p->psUI       = makePipe(p->dev, lib, @"uiVS", @"uiFS", BGRA, MTLPixelFormatInvalid, 1, 1, @"hud", err);
    if (!err.empty()) return false;

    // --- depth / sampler state -------------------------------------------
    auto mkDepth = [&](MTLCompareFunction f, BOOL write) {
        MTLDepthStencilDescriptor* d = [MTLDepthStencilDescriptor new];
        d.depthCompareFunction = f;
        d.depthWriteEnabled = write;
        return [p->dev newDepthStencilStateWithDescriptor:d];
    };
    p->dsOpaque      = mkDepth(MTLCompareFunctionLess, YES);
    p->dsTestNoWrite = mkDepth(MTLCompareFunctionLess, NO);
    p->dsAlways      = mkDepth(MTLCompareFunctionAlways, NO);
    p->dsShadow      = mkDepth(MTLCompareFunctionLess, YES);

    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.compareFunction = MTLCompareFunctionLess;   // enables hardware PCF
    p->shadowSampler = [p->dev newSamplerStateWithDescriptor:sd];

    // --- static geometry --------------------------------------------------
    auto mkBuf = [&](const void* data, size_t bytes, NSString* label) {
        id<MTLBuffer> b = [p->dev newBufferWithBytes:data length:bytes
                                             options:MTLResourceStorageModeShared];
        b.label = label;
        return b;
    };
    uploadTerrain(p, terrain);

    std::vector<MeshVertex> mv; std::vector<uint32_t> mi;
    buildMeshLibrary(mv, mi, p->ranges);
    // Fold baseVertex into the indices so every draw can use baseVertex 0.
    for (int m = 0; m < MESH_COUNT; m++) {
        for (uint32_t k = 0; k < p->ranges[m].indexCount; k++)
            mi[p->ranges[m].firstIndex + k] += (uint32_t)p->ranges[m].baseVertex;
        p->ranges[m].baseVertex = 0;
    }
    p->meshVB = mkBuf(mv.data(), mv.size() * sizeof(MeshVertex), @"meshVB");
    p->meshIB = mkBuf(mi.data(), mi.size() * sizeof(uint32_t), @"meshIB");

    // --- per-frame dynamic buffers ---------------------------------------
    for (int i = 0; i < kFramesInFlight; i++) {
        p->uniformBuf[i]  = [p->dev newBufferWithLength:sizeof(FrameUniforms) options:MTLResourceStorageModeShared];
        p->instanceBuf[i] = [p->dev newBufferWithLength:kMaxInstances * sizeof(InstanceData) options:MTLResourceStorageModeShared];
        p->billboardBuf[i]= [p->dev newBufferWithLength:kMaxBillboards * sizeof(Billboard) options:MTLResourceStorageModeShared];
        p->decalBuf[i]    = [p->dev newBufferWithLength:kMaxBillboards * sizeof(Billboard) options:MTLResourceStorageModeShared];
        p->uiBuf[i]       = [p->dev newBufferWithLength:kMaxUIVerts * sizeof(UIVertex) options:MTLResourceStorageModeShared];
    }

    // --- shadow map + minimap + font -------------------------------------
    MTLTextureDescriptor* smd =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:DEP
                                                           width:p->cfg.shadowRes
                                                          height:p->cfg.shadowRes mipmapped:NO];
    smd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    smd.storageMode = MTLStorageModePrivate;
    p->shadowMap = [p->dev newTextureWithDescriptor:smd];

    // --- material textures -------------------------------------------------
    {
        struct { const char* file; float bump; } kMats[4] = {
            {"ground.png",  3.0f},  // fine gravel: strong micro-relief
            {"cliff.png",   4.0f},  // layered rock: deep strata shadows
            {"armor.png",   2.2f},  // panel seams: crisp but shallow
            {"crystal.jpg", 3.4f},  // faceted ore: hard edges, flat faces
        };
        const int TS = 1024;
        const int NMAT = 4;
        RawImage imgs[NMAT];
        bool all = true;
        for (int i = 0; i < NMAT; i++) {
            std::string path = findAsset(kMats[i].file);
            if (path.empty() || !loadImageRGBA(path, TS, imgs[i]) ||
                imgs[i].w != TS || imgs[i].h != TS) { all = false; break; }
            normaliseMeanLuminance(imgs[i]);
        }
        if (all) {
            auto makeArray = [&](MTLPixelFormat fmt) {
                MTLTextureDescriptor* d = [MTLTextureDescriptor new];
                d.textureType = MTLTextureType2DArray;
                d.pixelFormat = fmt;
                d.width = TS; d.height = TS;
                d.arrayLength = NMAT;
                d.mipmapLevelCount = (NSUInteger)std::floor(std::log2((double)TS)) + 1;
                d.usage = MTLTextureUsageShaderRead;
                d.storageMode = MTLStorageModeShared;
                return [p->dev newTextureWithDescriptor:d];
            };
            // Albedo is colour, so sRGB: sampling then returns linear values that
            // the lighting maths can use directly. Normals are data, not colour.
            p->matAlbedo = makeArray(MTLPixelFormatRGBA8Unorm_sRGB);
            p->matNormal = makeArray(MTLPixelFormatRGBA8Unorm);
            std::vector<uint8_t> nrm;
            for (int i = 0; i < NMAT; i++) {
                [p->matAlbedo replaceRegion:MTLRegionMake2D(0, 0, TS, TS)
                                mipmapLevel:0 slice:i withBytes:imgs[i].px.data()
                                bytesPerRow:TS * 4 bytesPerImage:(NSUInteger)TS * TS * 4];
                deriveNormalHeight(imgs[i], kMats[i].bump, nrm);
                [p->matNormal replaceRegion:MTLRegionMake2D(0, 0, TS, TS)
                                mipmapLevel:0 slice:i withBytes:nrm.data()
                                bytesPerRow:TS * 4 bytesPerImage:(NSUInteger)TS * TS * 4];
            }
            p->texBytes += (size_t)TS * TS * 4 * NMAT * 2;
            p->texStrength = 1.0f;
        } else {
            fprintf(stderr, "starforge: assets/ textures not found, using procedural materials\n");
        }

        RawImage ex;
        std::string exPath = findAsset("explosion.jpg");
        if (exPath.empty()) exPath = findAsset("explosion.png");
        if (!exPath.empty() && loadImageRGBA(exPath, 1024, ex)) {
            MTLTextureDescriptor* d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB
                                                                   width:ex.w height:ex.h mipmapped:YES];
            d.usage = MTLTextureUsageShaderRead;
            d.storageMode = MTLStorageModeShared;
            p->explosionTex = [p->dev newTextureWithDescriptor:d];
            [p->explosionTex replaceRegion:MTLRegionMake2D(0, 0, ex.w, ex.h)
                               mipmapLevel:0 withBytes:ex.px.data() bytesPerRow:ex.w * 4];
            p->texBytes += (size_t)ex.w * ex.h * 4;
        }

        // Water ripples. The generator was asked for a greyscale *height* map
        // rather than a normal map: image models produce plausible-looking but
        // geometrically meaningless purple noise when asked for the latter, and
        // the Sobel pass here turns a height map into a correct normal anyway.
        {
            RawImage wh;
            std::string wp = findAsset("water-height.jpg");
            if (wp.empty()) wp = findAsset("water-height.png");
            if (!wp.empty() && loadImageRGBA(wp, 1024, wh)) {
                std::vector<uint8_t> nrm;
                // Far higher than the material textures use. Those are gravel
                // and rock, where neighbouring texels already differ sharply; a
                // smooth swell map has tiny one-pixel gradients, so the same
                // Sobel needs an order of magnitude more gain to produce a
                // surface normal you can actually see.
                deriveNormalHeight(wh, 26.0f, nrm);
                MTLTextureDescriptor* d =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                       width:wh.w height:wh.h mipmapped:YES];
                d.usage = MTLTextureUsageShaderRead;
                d.storageMode = MTLStorageModeShared;
                p->waterNrm = [p->dev newTextureWithDescriptor:d];
                [p->waterNrm replaceRegion:MTLRegionMake2D(0, 0, wh.w, wh.h)
                               mipmapLevel:0 withBytes:nrm.data() bytesPerRow:wh.w * 4];
                p->texBytes += (size_t)wh.w * wh.h * 4;
                p->waterTexStrength = 1.0f;
            }
        }

        // Scorch marks and smoke are coverage masks, so one channel is enough.
        auto loadMask = [&](const char* jpg, const char* png) -> id<MTLTexture> {
            RawImage img;
            std::string path = findAsset(jpg);
            if (path.empty()) path = findAsset(png);
            if (path.empty() || !loadImageRGBA(path, 1024, img)) return nil;
            std::vector<uint8_t> mask;
            toR8(img, mask);
            MTLTextureDescriptor* d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                   width:img.w height:img.h mipmapped:YES];
            d.usage = MTLTextureUsageShaderRead;
            d.storageMode = MTLStorageModeShared;
            id<MTLTexture> t = [p->dev newTextureWithDescriptor:d];
            [t replaceRegion:MTLRegionMake2D(0, 0, img.w, img.h)
                 mipmapLevel:0 withBytes:mask.data() bytesPerRow:img.w];
            p->texBytes += (size_t)img.w * img.h;
            return t;
        };
        p->scorchTex   = loadMask("scorch.jpg", "scorch.png");
        p->smokeTex    = loadMask("smoke.jpg",  "smoke.png");
        p->cloudTex    = loadMask("clouds.jpg", "clouds.png");
        p->particleTex = loadMask("particles.jpg", "particles.png");
        p->macroTex    = loadMask("terrain-macro.jpg", "terrain-macro.png");

        // One blit pass builds every mip chain.
        id<MTLCommandBuffer> cb = [p->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (p->matAlbedo)    [blit generateMipmapsForTexture:p->matAlbedo];
        if (p->matNormal)    [blit generateMipmapsForTexture:p->matNormal];
        if (p->explosionTex) [blit generateMipmapsForTexture:p->explosionTex];
        if (p->waterNrm)     [blit generateMipmapsForTexture:p->waterNrm];
        if (p->scorchTex)    [blit generateMipmapsForTexture:p->scorchTex];
        if (p->smokeTex)     [blit generateMipmapsForTexture:p->smokeTex];
        if (p->cloudTex)     [blit generateMipmapsForTexture:p->cloudTex];
        if (p->particleTex)  [blit generateMipmapsForTexture:p->particleTex];
        if (p->macroTex)     [blit generateMipmapsForTexture:p->macroTex];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        MTLSamplerDescriptor* rs = [MTLSamplerDescriptor new];
        rs.minFilter = MTLSamplerMinMagFilterLinear;
        rs.magFilter = MTLSamplerMinMagFilterLinear;
        rs.mipFilter = MTLSamplerMipFilterLinear;
        rs.sAddressMode = MTLSamplerAddressModeRepeat;
        rs.tAddressMode = MTLSamplerAddressModeRepeat;
        rs.maxAnisotropy = 4;   // terrain is viewed at a grazing angle
        p->repeatSampler = [p->dev newSamplerStateWithDescriptor:rs];
        if (p->texBytes)
            fprintf(stderr, "starforge: material textures %.1f MB resident (mips +33%%)\n",
                    p->texBytes / (1024.0 * 1024.0));
    }

    buildFontAtlas(p);

    // Per-pass GPU timing. The command buffer's own GPUStartTime/GPUEndTime span
    // the present wait, so it reports *more* time as the scene gets lighter --
    // useless for optimisation. Timestamp counters measure actual execution.
    if ([p->dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        for (id<MTLCounterSet> cs in [p->dev counterSets]) {
            if (![[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) continue;
            MTLCounterSampleBufferDescriptor* sd = [MTLCounterSampleBufferDescriptor new];
            sd.counterSet = cs;
            sd.sampleCount = 8;
            sd.storageMode = MTLStorageModeShared;
            p->countersOK = true;
            for (int i = 0; i < kFramesInFlight; i++) {
                NSError* ce = nil;
                p->gpuCounters[i] = [p->dev newCounterSampleBufferWithDescriptor:sd error:&ce];
                if (!p->gpuCounters[i]) p->countersOK = false;
            }
            // Metal's common timestamp counter reports nanoseconds on Apple GPUs.
            // (Verified against a known 60 Hz cadence: consecutive frames land
            // 1.38e7 units apart, i.e. 13.8 ms. Note this is a *different* clock
            // domain from -sampleTimestamps:gpuTimestamp:, which returns mach
            // ticks -- correlating the two directly is wrong.)
            p->gpuTickToMs = 1.0e-6;
            break;
        }
    }
    return true;
}

void Renderer::rebuildTerrain(const Terrain& terrain) { uploadTerrain(p_, terrain); }

RenderSettings Renderer::settings() const { return p_->cfg; }

void Renderer::applySettings(const RenderSettings& rs) {
    Impl* p = p_;
    const bool msaaChanged = (rs.msaa != p->cfg.msaa);
    const bool shadowChanged = (rs.shadowRes != p->cfg.shadowRes);
    p->cfg = rs;
    if (msaaChanged && p->lib) {
        std::string err;
        const MTLPixelFormat HDR = kHDRFormat;
        const MTLPixelFormat DEP = MTLPixelFormatDepth32Float;
        const NSUInteger n = (NSUInteger)std::max(1, rs.msaa);
        p->psSky      = makePipe(p->dev, p->lib, @"skyVS", @"skyFS", HDR, DEP, n, 0, @"sky", err);
        p->psTerrain  = makePipe(p->dev, p->lib, @"terrainVS", @"terrainFS", HDR, DEP, n, 0, @"terrain", err);
        p->psObject   = makePipe(p->dev, p->lib, @"objectVS", @"objectFS", HDR, DEP, n, 0, @"object", err);
        p->psWater    = makePipe(p->dev, p->lib, @"waterVS", @"waterFS", HDR, DEP, n, 1, @"water", err);
        p->psBillboard= makePipe(p->dev, p->lib, @"billboardVS", @"billboardFS", HDR, DEP, n, 2, @"particles", err);
        p->psDecal    = makePipe(p->dev, p->lib, @"billboardVS", @"decalFS",     HDR, DEP, n, 1, @"decals", err);
        int w = p->width, h = p->height;
        p->width = p->height = 0;      // force target rebuild
        resize(w, h);
    }
    if (shadowChanged) {
        MTLTextureDescriptor* smd =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                               width:rs.shadowRes
                                                              height:rs.shadowRes mipmapped:NO];
        smd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        smd.storageMode = MTLStorageModePrivate;
        p->shadowMap = [p->dev newTextureWithDescriptor:smd];
    }
}

void Renderer::resize(int w, int h) {
    Impl* p = p_;
    w = std::max(1, w); h = std::max(1, h);
    if (w == p->width && h == p->height && p->hdrResolve) return;
    p->width = w; p->height = h;

    auto mkTex = [&](MTLPixelFormat fmt, int tw, int th, int samples, bool memoryless, bool read) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor new];
        d.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
        d.pixelFormat = fmt;
        d.width = std::max(1, tw); d.height = std::max(1, th);
        d.sampleCount = samples;
        d.usage = MTLTextureUsageRenderTarget | (read ? MTLTextureUsageShaderRead : 0);
        // MSAA targets never leave tile memory on Apple GPUs; they resolve on store.
        d.storageMode = memoryless ? MTLStorageModeMemoryless : MTLStorageModePrivate;
        return [p->dev newTextureWithDescriptor:d];
    };
    const MTLPixelFormat HDR = kHDRFormat;
    const int ms = std::max(1, p->cfg.msaa);
    p->hdrMSAA   = mkTex(HDR, w, h, ms, true, false);
    p->depthMSAA = mkTex(MTLPixelFormatDepth32Float, w, h, ms, true, false);
    p->hdrResolve= mkTex(HDR, w, h, 1, false, true);
    p->bloomHalfA= mkTex(HDR, w / 2, h / 2, 1, false, true);
    p->bloomHalfB= mkTex(HDR, w / 2, h / 2, 1, false, true);
    p->bloomQtrA = mkTex(HDR, w / 4, h / 4, 1, false, true);
    p->bloomQtrB = mkTex(HDR, w / 4, h / 4, 1, false, true);
}

static void writePNG(id<MTLTexture> tex, const std::string& path) {
    const int w = (int)tex.width, h = (int)tex.height;
    std::vector<uint8_t> buf((size_t)w * h * 4);
    [tex getBytes:buf.data() bytesPerRow:w * 4
       fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    for (size_t i = 0; i < buf.size(); i += 4) std::swap(buf[i], buf[i + 2]);  // BGRA -> RGBA

    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(buf.data(), w, h, 8, w * 4, cs,
                                             kCGImageAlphaNoneSkipLast);
    CGImageRef img = CGBitmapContextCreateImage(ctx);
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        NULL, (const UInt8*)path.c_str(), path.size(), false);
    CGImageDestinationRef dst = CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, NULL);
    if (!dst) {
        fprintf(stderr, "capture: cannot open '%s' for writing\n", path.c_str());
    } else {
        CGImageDestinationAddImage(dst, img, NULL);
        if (!CGImageDestinationFinalize(dst))
            fprintf(stderr, "capture: PNG encode failed for '%s'\n", path.c_str());
        CFRelease(dst);
    }
    CFRelease(url); CGImageRelease(img); CGContextRelease(ctx); CGColorSpaceRelease(cs);
}

void Renderer::requestCapture(const std::string& path) {
    p_->capturePath = path;
    p_->captureDone = false;
}
bool Renderer::captureComplete() const { return p_->captureDone; }

// Marks one pass in the timestamp buffer. Either boundary may be skipped, which
// is how the six bloom passes are timed as a single span.
static void sampleAt(MTLRenderPassDescriptor* rp, id<MTLCounterSampleBuffer> buf,
                     NSUInteger startIdx, NSUInteger endIdx) {
    if (!buf) return;
    MTLRenderPassSampleBufferAttachmentDescriptor* a = rp.sampleBufferAttachments[0];
    a.sampleBuffer = buf;
    a.startOfVertexSampleIndex = startIdx;
    a.endOfFragmentSampleIndex = endIdx;
}

void Renderer::render(const RenderFrame& f) {
    Impl* p = p_;
    if (!p->hdrResolve) return;
    dispatch_semaphore_wait(p->sem, DISPATCH_TIME_FOREVER);
    const int fi = p->frameIndex % kFramesInFlight;
    p->frameIndex++;

    // ---- shadow cascade fitted around the camera's focus point -----------
    const float R = 105.0f;
    v3 L = normalize(f.sunDir);
    v3 up = std::fabs(L.y) > 0.95f ? v3{0, 0, 1} : v3{0, 1, 0};
    m4 lightView = lookAt(f.shadowFocus + L * 260.0f, f.shadowFocus, up);
    // Snap the cascade to whole texels so shadow edges don't crawl when panning.
    v3 cLS = transformPoint(lightView, f.shadowFocus);
    const float texelWorld = 2.0f * R / (float)p->cfg.shadowRes;
    float sx = std::round(cLS.x / texelWorld) * texelWorld - cLS.x;
    float sy = std::round(cLS.y / texelWorld) * texelWorld - cLS.y;
    m4 lightProj = ortho(-R + sx, R + sx, -R + sy, R + sy, 1.0f, 520.0f);
    m4 lightVP = lightProj * lightView;

    // ---- upload per-frame data -------------------------------------------
    FrameUniforms U{};
    U.viewProj = f.viewProj;
    U.view = f.view;
    U.lightViewProj = lightVP;
    U.invViewProj = f.invViewProj;
    U.camPos = v4(f.camPos, f.time);
    U.sunDir = v4(L, f.sunIntensity);
    U.sunColor = v4(f.sunColor, f.ambient);
    U.fog = v4(f.fogColor, f.fogDensity);
    U.misc = v4{f.waterLevel, 1.0f / (float)p->cfg.shadowRes, p->texStrength, 1200.0f};

    // ---- fog of war ------------------------------------------------------
    float fowStrength = 0.0f;
    if (f.fowN > 0 && f.fowStrength > 0.0f &&
        f.fow.size() >= (size_t)f.fowN * f.fowN * 2 && f.mapSize > 0.0f) {
        if (p->fowN != f.fowN) {
            MTLTextureDescriptor* d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG8Unorm
                                                                   width:f.fowN height:f.fowN
                                                               mipmapped:NO];
            d.usage = MTLTextureUsageShaderRead;
            d.storageMode = MTLStorageModeShared;
            p->fowTex = [p->dev newTextureWithDescriptor:d];
            p->fowTex.label = @"fogOfWar";
            p->fowN = f.fowN;
        }
        [p->fowTex replaceRegion:MTLRegionMake2D(0, 0, f.fowN, f.fowN)
                     mipmapLevel:0 withBytes:f.fow.data() bytesPerRow:f.fowN * 2];
        fowStrength = f.fowStrength;
        U.texParams.x = 1.0f / f.mapSize;
    }
    U.texParams.y = p->fowTex ? fowStrength : 0.0f;
    U.texParams.z = p->waterTexStrength;
    U.texFlags = v4{p->cloudTex ? 1.0f : 0.0f, p->particleTex ? 1.0f : 0.0f,
                    p->macroTex ? 1.0f : 0.0f, 0.0f};
    memcpy([p->uniformBuf[fi] contents], &U, sizeof(U));

    uint32_t instOffset[MESH_COUNT] = {0};
    uint32_t instCount[MESH_COUNT] = {0};
    {
        auto* dst = (InstanceData*)[p->instanceBuf[fi] contents];
        uint32_t off = 0;
        for (int m = 0; m < MESH_COUNT; m++) {
            uint32_t n = (uint32_t)f.instances[m].size();
            if (off + n > kMaxInstances) n = kMaxInstances > off ? kMaxInstances - off : 0;
            instOffset[m] = off; instCount[m] = n;
            if (n) memcpy(dst + off, f.instances[m].data(), n * sizeof(InstanceData));
            off += n;
        }
    }
    uint32_t bbCount = (uint32_t)std::min<size_t>(f.billboards.size(), kMaxBillboards);
    if (bbCount) memcpy([p->billboardBuf[fi] contents], f.billboards.data(), bbCount * sizeof(Billboard));
    uint32_t dcCount = (uint32_t)std::min<size_t>(f.decals.size(), kMaxBillboards);
    if (dcCount) memcpy([p->decalBuf[fi] contents], f.decals.data(), dcCount * sizeof(Billboard));

    uint32_t uiCount = (uint32_t)std::min<size_t>(f.uiVerts.size(), kMaxUIVerts);
    uint32_t mmCount = (uint32_t)std::min<size_t>(f.minimapVerts.size(), kMaxUIVerts - uiCount);
    {
        auto* dst = (UIVertex*)[p->uiBuf[fi] contents];
        if (mmCount) memcpy(dst, f.minimapVerts.data(), mmCount * sizeof(UIVertex));
        if (uiCount) memcpy(dst + mmCount, f.uiVerts.data(), uiCount * sizeof(UIVertex));
    }

    id<CAMetalDrawable> drawable = [p->layer nextDrawable];
    if (!drawable) { dispatch_semaphore_signal(p->sem); return; }
    id<MTLCommandBuffer> cb = [p->queue commandBuffer];
    RenderStats st{};

    // ---- pass 1: shadow ---------------------------------------------------
    {
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.depthAttachment.texture = p->shadowMap;
        rp.depthAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        rp.depthAttachment.clearDepth = 1.0;
        sampleAt(rp, p->gpuCounters[fi], 0, 1);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        enc.label = @"shadow";
        [enc setDepthStencilState:p->dsShadow];
        [enc setFrontFacingWinding:MTLWindingCounterClockwise];
        [enc setCullMode:MTLCullModeNone];
        // Bias is in normalised depth over a ~519 unit frustum, so the clamp must
        // stay tiny: 0.02 would be 10 world units and would push buildings clean
        // out of their own shadows (steep depth slopes always hit the clamp).
        [enc setDepthBias:0.8f slopeScale:1.6f clamp:0.0025f];

        [enc setRenderPipelineState:p->psTerrainShadow];
        [enc setVertexBuffer:p->terrainVB offset:0 atIndex:0];
        [enc setVertexBuffer:p->uniformBuf[fi] offset:0 atIndex:1];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:p->terrainIdxCount
                         indexType:MTLIndexTypeUInt32 indexBuffer:p->terrainIB indexBufferOffset:0];
        st.drawCalls++;

        [enc setRenderPipelineState:p->psObjectShadow];
        [enc setVertexBuffer:p->meshVB offset:0 atIndex:0];
        [enc setVertexBuffer:p->instanceBuf[fi] offset:0 atIndex:2];
        for (int m = 0; m < MESH_COUNT; m++) {
            if (!instCount[m] || m == MESH_SEL_RING) continue;
            [enc setVertexBufferOffset:instOffset[m] * sizeof(InstanceData) atIndex:2];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:p->ranges[m].indexCount
                             indexType:MTLIndexTypeUInt32 indexBuffer:p->meshIB
                     indexBufferOffset:p->ranges[m].firstIndex * sizeof(uint32_t)
                         instanceCount:instCount[m]];
            st.drawCalls++;
        }
        [enc endEncoding];
    }

    // ---- pass 2: main scene ----------------------------------------------
    {
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        const bool msaaOn = p->cfg.msaa > 1;
        rp.colorAttachments[0].texture = msaaOn ? p->hdrMSAA : p->hdrResolve;
        if (msaaOn) rp.colorAttachments[0].resolveTexture = p->hdrResolve;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction =
            msaaOn ? MTLStoreActionMultisampleResolve : MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(f.fogColor.x, f.fogColor.y, f.fogColor.z, 1.0);
        rp.depthAttachment.texture = p->depthMSAA;
        rp.depthAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.storeAction = MTLStoreActionDontCare;
        rp.depthAttachment.clearDepth = 1.0;
        sampleAt(rp, p->gpuCounters[fi], 2, 3);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        enc.label = @"scene";
        [enc setFrontFacingWinding:MTLWindingCounterClockwise];
        [enc setFragmentBuffer:p->uniformBuf[fi] offset:0 atIndex:1];
        [enc setFragmentTexture:p->shadowMap atIndex:0];
        [enc setFragmentSamplerState:p->shadowSampler atIndex:0];
        [enc setFragmentTexture:p->matAlbedo atIndex:1];
        [enc setFragmentTexture:p->matNormal atIndex:2];
        [enc setFragmentTexture:p->explosionTex atIndex:3];
        [enc setFragmentTexture:p->fowTex atIndex:4];
        [enc setFragmentTexture:p->scorchTex atIndex:5];
        [enc setFragmentTexture:p->smokeTex atIndex:6];
        [enc setFragmentTexture:p->waterNrm atIndex:7];
        [enc setFragmentTexture:p->cloudTex atIndex:8];
        [enc setFragmentTexture:p->particleTex atIndex:9];
        [enc setFragmentTexture:p->macroTex atIndex:10];
        [enc setFragmentSamplerState:p->repeatSampler atIndex:1];

        // sky
        [enc setRenderPipelineState:p->psSky];
        [enc setDepthStencilState:p->dsAlways];
        [enc setCullMode:MTLCullModeNone];
        [enc setFragmentBuffer:p->uniformBuf[fi] offset:0 atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        st.drawCalls++;

        // terrain
        [enc setRenderPipelineState:p->psTerrain];
        [enc setDepthStencilState:p->dsOpaque];
        [enc setCullMode:MTLCullModeBack];
        [enc setVertexBuffer:p->terrainVB offset:0 atIndex:0];
        [enc setVertexBuffer:p->uniformBuf[fi] offset:0 atIndex:1];
        [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:p->terrainIdxCount
                         indexType:MTLIndexTypeUInt32 indexBuffer:p->terrainIB indexBufferOffset:0];
        st.drawCalls++;
        st.triangles += p->terrainIdxCount / 3;

        // instanced objects
        [enc setRenderPipelineState:p->psObject];
        [enc setVertexBuffer:p->meshVB offset:0 atIndex:0];
        [enc setVertexBuffer:p->instanceBuf[fi] offset:0 atIndex:2];
        for (int m = 0; m < MESH_COUNT; m++) {
            if (!instCount[m]) continue;
            // slice, object-space uv scale, detail blend, normal strength.
            // Crystals get the faceted ore texture and boulders the cliff
            // strata; everything built by a faction gets panel plating.
            v4 objMat = (m == MESH_ORE) ? v4{3.0f, 0.55f, 0.55f, 1.10f}
                      : (m == MESH_BOULDER)    ? v4{1.0f, 0.30f, 0.70f, 1.00f}
                                            : v4{2.0f, 0.42f, 0.68f, 0.85f};
            [enc setFragmentBytes:&objMat length:sizeof(v4) atIndex:3];
            // Ground rings are flat decals; culling them by face would drop them.
            [enc setCullMode:(m == MESH_SEL_RING) ? MTLCullModeNone : MTLCullModeBack];
            [enc setVertexBufferOffset:instOffset[m] * sizeof(InstanceData) atIndex:2];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:p->ranges[m].indexCount
                             indexType:MTLIndexTypeUInt32 indexBuffer:p->meshIB
                     indexBufferOffset:p->ranges[m].firstIndex * sizeof(uint32_t)
                         instanceCount:instCount[m]];
            st.drawCalls++;
            st.instances += instCount[m];
            st.triangles += (p->ranges[m].indexCount / 3) * instCount[m];
        }

        // water
        if (p->waterIdxCount) {
            [enc setRenderPipelineState:p->psWater];
            [enc setDepthStencilState:p->dsTestNoWrite];
            [enc setCullMode:MTLCullModeNone];
            [enc setVertexBuffer:p->waterVB offset:0 atIndex:0];
            [enc setVertexBuffer:p->uniformBuf[fi] offset:0 atIndex:1];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:p->waterIdxCount
                             indexType:MTLIndexTypeUInt32 indexBuffer:p->waterIB indexBufferOffset:0];
            st.drawCalls++;
        }

        // alpha-blended decals: scorch first, then smoke above it
        if (dcCount) {
            [enc setRenderPipelineState:p->psDecal];
            [enc setDepthStencilState:p->dsTestNoWrite];
            [enc setCullMode:MTLCullModeNone];
            [enc setVertexBuffer:p->decalBuf[fi] offset:0 atIndex:0];
            [enc setVertexBuffer:p->uniformBuf[fi] offset:0 atIndex:1];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6
                  instanceCount:dcCount];
            st.drawCalls++;
        }

        // particles
        if (bbCount) {
            [enc setRenderPipelineState:p->psBillboard];
            [enc setDepthStencilState:p->dsTestNoWrite];
            [enc setCullMode:MTLCullModeNone];
            [enc setVertexBuffer:p->billboardBuf[fi] offset:0 atIndex:0];
            [enc setVertexBuffer:p->uniformBuf[fi] offset:0 atIndex:1];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6
                  instanceCount:bbCount];
            st.drawCalls++;
        }
        [enc endEncoding];
    }

    // ---- pass 3: bloom chain ---------------------------------------------
    int bloomPassIdx = 0;
    auto blitPass = [&](id<MTLTexture> dstTex, id<MTLRenderPipelineState> ps,
                        id<MTLTexture> src0, id<MTLTexture> src1, id<MTLTexture> src2,
                        v4 params, NSString* label) {
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = dstTex;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        // Time the whole bloom chain as one span: start on the first pass, end on
        // the last, with the boundaries in between left unsampled.
        if (bloomPassIdx == 0) sampleAt(rp, p->gpuCounters[fi], 4, MTLCounterDontSample);
        if (bloomPassIdx == 5) sampleAt(rp, p->gpuCounters[fi], MTLCounterDontSample, 5);
        bloomPassIdx++;
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        enc.label = label;
        [enc setRenderPipelineState:ps];
        [enc setFragmentTexture:src0 atIndex:0];
        if (src1) [enc setFragmentTexture:src1 atIndex:1];
        if (src2) [enc setFragmentTexture:src2 atIndex:2];
        [enc setFragmentBytes:&params length:sizeof(v4) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        st.drawCalls++;
    };

    const float hw = 1.0f / std::max(1, p->width / 2), hh = 1.0f / std::max(1, p->height / 2);
    const float qw = 1.0f / std::max(1, p->width / 4), qh = 1.0f / std::max(1, p->height / 4);
    if (!p->cfg.bloom) bloomPassIdx = 6;   // skip the chain entirely
    if (p->cfg.bloom) {
    blitPass(p->bloomHalfA, p->psBright, p->hdrResolve, nil, nil,
             v4{f.bloomThreshold, 1.0f, 1.0f / p->width, 1.0f / p->height}, @"bright");
    blitPass(p->bloomHalfB, p->psBlur, p->bloomHalfA, nil, nil, v4{hw * 1.2f, 0, 0, 0}, @"blurH");
    blitPass(p->bloomHalfA, p->psBlur, p->bloomHalfB, nil, nil, v4{0, hh * 1.2f, 0, 0}, @"blurV");
    blitPass(p->bloomQtrA, p->psDown, p->bloomHalfA, nil, nil, v4{0, 0, hw, hh}, @"down");
    blitPass(p->bloomQtrB, p->psBlur, p->bloomQtrA, nil, nil, v4{qw * 1.4f, 0, 0, 0}, @"blurH2");
    blitPass(p->bloomQtrA, p->psBlur, p->bloomQtrB, nil, nil, v4{0, qh * 1.4f, 0, 0}, @"blurV2");
    }

    // ---- pass 4: tonemap + HUD -------------------------------------------
    {
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = drawable.texture;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        sampleAt(rp, p->gpuCounters[fi], 6, 7);
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
        enc.label = @"composite+hud";
        [enc setRenderPipelineState:p->psComposite];
        [enc setFragmentTexture:p->hdrResolve atIndex:0];
        [enc setFragmentTexture:p->bloomHalfA atIndex:1];
        [enc setFragmentTexture:p->bloomQtrA atIndex:2];
        v4 cp{f.exposure, p->cfg.bloom ? f.bloomIntensity : 0.0f, 0.0016f, f.time};
        [enc setFragmentBytes:&cp length:sizeof(v4) atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        st.drawCalls++;

        if (uiCount || mmCount) {
            v4 screen{1.0f / p->width, 1.0f / p->height, 0, 0};
            [enc setRenderPipelineState:p->psUI];
            [enc setVertexBuffer:p->uiBuf[fi] offset:0 atIndex:0];
            [enc setVertexBytes:&screen length:sizeof(v4) atIndex:1];
            if (mmCount) {
                [enc setFragmentTexture:p->minimapTex atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:mmCount];
                st.drawCalls++;
            }
            if (uiCount) {
                [enc setFragmentTexture:p->fontTex atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:mmCount vertexCount:uiCount];
                st.drawCalls++;
            }
        }
        [enc endEncoding];
    }

    // Bind the pending capture to *this* command buffer: an earlier frame's
    // completion handler must not consume a request it never encoded a blit for.
    const bool doCapture = !p->capturePath.empty();
    const std::string capturePath = p->capturePath;
    if (doCapture) {
        p->capturePath.clear();
        if (!p->captureTex || (int)p->captureTex.width != p->width ||
            (int)p->captureTex.height != p->height) {
            MTLTextureDescriptor* cd =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                   width:p->width height:p->height
                                                               mipmapped:NO];
            cd.usage = MTLTextureUsageShaderRead;
            cd.storageMode = MTLStorageModeShared;
            p->captureTex = [p->dev newTextureWithDescriptor:cd];
        }
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:drawable.texture sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(p->width, p->height, 1)
                    toTexture:p->captureTex destinationSlice:0 destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
    }

    [cb presentDrawable:drawable];
    __block dispatch_semaphore_t sem = p->sem;
    __block Impl* ip = p;
    __block RenderStats fst = st;
    __block id<MTLCounterSampleBuffer> ctr = p->countersOK ? p->gpuCounters[fi] : nil;
    const double tickMs = p->gpuTickToMs;   // captured by value; no __block needed
    fst.texMB = p->texBytes / (1024.0 * 1024.0);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> buf) {
        fst.gpuMs = (buf.GPUEndTime - buf.GPUStartTime) * 1000.0;
        if (ctr) {
            NSData* d = [ctr resolveCounterRange:NSMakeRange(0, 8)];
            if (d && d.length >= 8 * sizeof(MTLCounterResultTimestamp)) {
                const MTLCounterResultTimestamp* t =
                    (const MTLCounterResultTimestamp*)d.bytes;
                auto ms = [&](int a, int b) {
                    if (t[a].timestamp == MTLCounterErrorValue ||
                        t[b].timestamp == MTLCounterErrorValue) return 0.0;
                    if (t[b].timestamp <= t[a].timestamp) return 0.0;
                    return (double)(t[b].timestamp - t[a].timestamp) * tickMs;
                };
                fst.shadowMs    = ms(0, 1);
                fst.sceneMs     = ms(2, 3);
                fst.bloomMs     = ms(4, 5);
                fst.compositeMs = ms(6, 7);
                // Shadow through bloom only. The composite pass writes to the
                // drawable, so its span absorbs the GPU-side wait for the display
                // to release it -- including it made lighter frames measure
                // *slower*, which is the opposite of useful.
                fst.gpuFrameMs  = ms(0, 5);
                if (fst.gpuFrameMs <= 0.0) fst.gpuFrameMs = ms(0, 3);   // bloom disabled
            }
        }
        ip->stats = fst;
        if (doCapture) {
            writePNG(ip->captureTex, capturePath);
            ip->captureDone = true;
        }
        dispatch_semaphore_signal(sem);
    }];
    [cb commit];
}

// ---------------------------------------------------------------- HUD helpers
const MeshRange& Renderer::meshRange(int id) const { return p_->ranges[id]; }
RenderStats Renderer::stats() const { return p_->stats; }
float Renderer::lineHeight(float size) const { return p_->cellH * size; }

float Renderer::textWidth(const char* s, float size) const {
    float w = 0;
    for (const char* c = s; *c; ++c) {
        int i = (int)(unsigned char)*c - 32;
        if (i < 0 || i > 94) i = 0;
        w += p_->glyphs[i].advance * size;
    }
    return w;
}

static inline void pushQuad(std::vector<UIVertex>& out, float x, float y, float w, float h,
                            float u0, float v0, float u1, float v1, v4 c) {
    UIVertex a{x, y, u0, v0, c.x, c.y, c.z, c.w};
    UIVertex b{x + w, y, u1, v0, c.x, c.y, c.z, c.w};
    UIVertex d{x + w, y + h, u1, v1, c.x, c.y, c.z, c.w};
    UIVertex e{x, y + h, u0, v1, c.x, c.y, c.z, c.w};
    out.push_back(a); out.push_back(b); out.push_back(d);
    out.push_back(a); out.push_back(d); out.push_back(e);
}

void Renderer::text(std::vector<UIVertex>& out, const char* s, float x, float y,
                    float size, v4 color) const {
    const float cw = p_->cellW * size, ch = p_->cellH * size, px = p_->padX * size;
    float pen = x;
    for (const char* c = s; *c; ++c) {
        int i = (int)(unsigned char)*c - 32;
        if (i < 0 || i > 94) i = 0;
        const Glyph& g = p_->glyphs[i];
        if (*c != ' ') pushQuad(out, pen - px, y, cw, ch, g.u0, g.v0, g.u1, g.v1, color);
        pen += g.advance * size;
    }
}

void Renderer::rect(std::vector<UIVertex>& out, float x, float y, float w, float h, v4 c) const {
    const Glyph& g = p_->glyphs[kWhiteGlyph];
    pushQuad(out, x, y, w, h, g.u0, g.v0, g.u1, g.v1, c);
}

void Renderer::frame(std::vector<UIVertex>& out, float x, float y, float w, float h,
                     float t, v4 c) const {
    rect(out, x, y, w, t, c);
    rect(out, x, y + h - t, w, t, c);
    rect(out, x, y + t, t, h - 2 * t, c);
    rect(out, x + w - t, y + t, t, h - 2 * t, c);
}

void Renderer::minimapQuad(std::vector<UIVertex>& out, float x, float y, float w, float h,
                           v4 tint) const {
    pushQuad(out, x, y, w, h, 0.0f, 0.0f, 1.0f, 1.0f, tint);
}

} // namespace sf
