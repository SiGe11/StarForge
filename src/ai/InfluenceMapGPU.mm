// InfluenceMapGPU.mm — Metal compute backend for the AI's spatial fields.
//
// Influence is the one part of this AI that is genuinely a wide parallel
// arithmetic problem: every grid cell accumulates a falloff term from every
// unit. At 64x64 cells against a few hundred units that is millions of
// multiply-adds several times a second, which is exactly what hardware.md
// recommends pushing onto the GPU while symbolic reasoning stays on the CPU.
#import <Metal/Metal.h>
#include "AI.h"
#include <string>

namespace sf::ai {

namespace {

const char* kInfluenceMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct IUnit { float x, z, strength, range, team, p0, p1, p2; };

kernel void influenceKernel(device const IUnit*  units  [[buffer(0)]],
                            constant uint&       count  [[buffer(1)]],
                            constant float2&     params [[buffer(2)]],  // gridN, cellSize
                            device float*        out    [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint N = (uint)params.x;
    if (gid.x >= N || gid.y >= N) return;
    float cell = params.y;
    float2 p = float2((float(gid.x) + 0.5f) * cell, (float(gid.y) + 0.5f) * cell);

    float friendly = 0.0f, enemy = 0.0f;
    for (uint i = 0; i < count; i++) {
        IUnit u = units[i];
        float2 d = p - float2(u.x, u.z);
        float d2 = dot(d, d);
        float reach = u.range * 2.0f;
        if (d2 > reach * reach) continue;
        float f = u.strength / (1.0f + d2 / (u.range * u.range));
        if (u.team > 0.0f) friendly += f; else enemy += f;
    }
    uint idx = (gid.y * N + gid.x) * 4u;
    out[idx + 0] = friendly;
    out[idx + 1] = enemy;
    out[idx + 2] = enemy;     // threat currently mirrors enemy influence
    // channel 3 (information age) is filled on the CPU: only the game knows it
}
)MSL";

struct MetalInfluenceBackend : IInfluenceBackend {
    id<MTLDevice>               dev = nil;
    id<MTLCommandQueue>         queue = nil;
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer>               unitBuf = nil;
    id<MTLBuffer>               outBuf = nil;
    NSUInteger                  unitCap = 0, outCap = 0;

    const char* name() const override { return "metal-gpu"; }

    bool compute(const InfluenceUnit* units, int n, int gridN, float cellSize,
                 float* out) override {
        if (n <= 0 || !pso) return false;
        @autoreleasepool {
            const NSUInteger needUnits = (NSUInteger)n * sizeof(InfluenceUnit);
            if (needUnits > unitCap) {
                unitCap = needUnits * 2;
                unitBuf = [dev newBufferWithLength:unitCap options:MTLResourceStorageModeShared];
            }
            const NSUInteger needOut = (NSUInteger)gridN * gridN * 4 * sizeof(float);
            if (needOut > outCap) {
                outCap = needOut;
                outBuf = [dev newBufferWithLength:outCap options:MTLResourceStorageModeShared];
            }
            if (!unitBuf || !outBuf) return false;
            memcpy([unitBuf contents], units, needUnits);

            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:unitBuf offset:0 atIndex:0];
            uint32_t count = (uint32_t)n;
            [enc setBytes:&count length:sizeof(count) atIndex:1];
            float params[2] = {(float)gridN, cellSize};
            [enc setBytes:params length:sizeof(params) atIndex:2];
            [enc setBuffer:outBuf offset:0 atIndex:3];

            NSUInteger w = pso.threadExecutionWidth;
            NSUInteger h = std::max<NSUInteger>(1, pso.maxTotalThreadsPerThreadgroup / w);
            [enc dispatchThreads:MTLSizeMake(gridN, gridN, 1)
           threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) return false;

            // Unified memory: the result is already CPU-visible, no blit needed.
            const float* src = (const float*)[outBuf contents];
            const int cells = gridN * gridN;
            for (int i = 0; i < cells; i++) {
                out[i * 4 + 0] = src[i * 4 + 0];
                out[i * 4 + 1] = src[i * 4 + 1];
                out[i * 4 + 2] = src[i * 4 + 2];
            }
        }
        return true;
    }
};

} // namespace

IInfluenceBackend* createMetalInfluenceBackend(std::string& err) {
    @autoreleasepool {
        auto* b = new MetalInfluenceBackend();
        b->dev = MTLCreateSystemDefaultDevice();
        if (!b->dev) { err = "no Metal device"; delete b; return nullptr; }
        b->queue = [b->dev newCommandQueue];

        NSError* e = nil;
        id<MTLLibrary> lib =
            [b->dev newLibraryWithSource:[NSString stringWithUTF8String:kInfluenceMSL]
                                 options:nil error:&e];
        if (!lib) {
            err = std::string("influence shader: ") + [[e localizedDescription] UTF8String];
            delete b; return nullptr;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"influenceKernel"];
        if (!fn) { err = "influenceKernel missing"; delete b; return nullptr; }
        b->pso = [b->dev newComputePipelineStateWithFunction:fn error:&e];
        if (!b->pso) {
            err = std::string("influence pipeline: ") + [[e localizedDescription] UTF8String];
            delete b; return nullptr;
        }
        return b;
    }
}

} // namespace sf::ai
