// Math.h — minimal column-major linear algebra for the renderer and simulation.
#pragma once
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace sf {

constexpr float PI  = 3.14159265358979323846f;
constexpr float TAU = 6.28318530717958647692f;
constexpr float DEG = PI / 180.0f;

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float saturate(float v) { return clampf(v, 0.0f, 1.0f); }
inline float smoothstepf(float e0, float e1, float x) {
    float t = saturate((x - e0) / (e1 - e0 + 1e-9f));
    return t * t * (3.0f - 2.0f * t);
}
inline float signf(float v) { return v < 0.0f ? -1.0f : (v > 0.0f ? 1.0f : 0.0f); }

// ---------------------------------------------------------------- vec2
struct v2 {
    float x = 0, y = 0;
    v2() = default;
    v2(float x_, float y_) : x(x_), y(y_) {}
};
inline v2 operator+(v2 a, v2 b) { return {a.x + b.x, a.y + b.y}; }
inline v2 operator-(v2 a, v2 b) { return {a.x - b.x, a.y - b.y}; }
inline v2 operator*(v2 a, float s) { return {a.x * s, a.y * s}; }
inline v2 operator*(float s, v2 a) { return {a.x * s, a.y * s}; }
inline v2 operator/(v2 a, float s) { return {a.x / s, a.y / s}; }
inline v2& operator+=(v2& a, v2 b) { a.x += b.x; a.y += b.y; return a; }
inline v2& operator-=(v2& a, v2 b) { a.x -= b.x; a.y -= b.y; return a; }
inline v2& operator*=(v2& a, float s) { a.x *= s; a.y *= s; return a; }
inline float dot(v2 a, v2 b) { return a.x * b.x + a.y * b.y; }
inline float cross(v2 a, v2 b) { return a.x * b.y - a.y * b.x; }
inline float length(v2 a) { return std::sqrt(a.x * a.x + a.y * a.y); }
inline float length2(v2 a) { return a.x * a.x + a.y * a.y; }
inline v2 normalize(v2 a) { float l = length(a); return l > 1e-8f ? a / l : v2{0, 0}; }
inline v2 perp(v2 a) { return {-a.y, a.x}; }
inline v2 lerp(v2 a, v2 b, float t) { return a + (b - a) * t; }

// ---------------------------------------------------------------- vec3
struct v3 {
    float x = 0, y = 0, z = 0;
    v3() = default;
    v3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    explicit v3(float s) : x(s), y(s), z(s) {}
};
inline v3 operator+(v3 a, v3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline v3 operator-(v3 a, v3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline v3 operator-(v3 a) { return {-a.x, -a.y, -a.z}; }
inline v3 operator*(v3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline v3 operator*(float s, v3 a) { return {a.x * s, a.y * s, a.z * s}; }
inline v3 operator*(v3 a, v3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline v3 operator/(v3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline v3& operator+=(v3& a, v3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline v3& operator-=(v3& a, v3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
inline v3& operator*=(v3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
inline float dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline v3 cross(v3 a, v3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(v3 a) { return std::sqrt(dot(a, a)); }
inline float length2(v3 a) { return dot(a, a); }
inline v3 normalize(v3 a) { float l = length(a); return l > 1e-8f ? a / l : v3{0, 0, 0}; }
inline v3 lerp(v3 a, v3 b, float t) { return a + (b - a) * t; }
inline v3 vmin(v3 a, v3 b) { return {std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z)}; }
inline v3 vmax(v3 a, v3 b) { return {std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z)}; }
// The sim is 2D on the XZ plane; Y is up.
inline v2 xz(v3 a) { return {a.x, a.z}; }
inline v3 xz2v3(v2 a, float y = 0.0f) { return {a.x, y, a.y}; }

// ---------------------------------------------------------------- vec4
struct v4 {
    float x = 0, y = 0, z = 0, w = 0;
    v4() = default;
    v4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    v4(v3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    v3 xyz() const { return {x, y, z}; }
};
inline v4 operator+(v4 a, v4 b) { return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
inline v4 operator*(v4 a, float s) { return {a.x*s, a.y*s, a.z*s, a.w*s}; }
inline float dot(v4 a, v4 b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
inline v4 lerp(v4 a, v4 b, float t) { return a + (b + a * -1.0f) * t; }

// ---------------------------------------------------------------- mat4
// Column-major, matching Metal's float4x4. m.c[i] is column i.
struct m4 {
    v4 c[4];
    m4() : c{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}} {}
    static m4 identity() { return m4(); }
    static m4 zero() { m4 m; for (int i = 0; i < 4; i++) m.c[i] = v4{0,0,0,0}; return m; }
};

inline v4 operator*(const m4& m, v4 v) {
    return m.c[0] * v.x + m.c[1] * v.y + m.c[2] * v.z + m.c[3] * v.w;
}
inline m4 operator*(const m4& a, const m4& b) {
    m4 r;
    for (int i = 0; i < 4; i++) r.c[i] = a * b.c[i];
    return r;
}
inline v3 transformPoint(const m4& m, v3 p) { return (m * v4(p, 1.0f)).xyz(); }
inline v3 transformDir(const m4& m, v3 d) { return (m * v4(d, 0.0f)).xyz(); }

inline m4 translate(v3 t) {
    m4 m; m.c[3] = v4(t, 1.0f); return m;
}
inline m4 scale(v3 s) {
    m4 m; m.c[0].x = s.x; m.c[1].y = s.y; m.c[2].z = s.z; return m;
}
inline m4 rotateY(float a) {
    float s = std::sin(a), c = std::cos(a);
    m4 m; m.c[0] = {c, 0, -s, 0}; m.c[2] = {s, 0, c, 0}; return m;
}
inline m4 rotateX(float a) {
    float s = std::sin(a), c = std::cos(a);
    m4 m; m.c[1] = {0, c, s, 0}; m.c[2] = {0, -s, c, 0}; return m;
}
inline m4 rotateZ(float a) {
    float s = std::sin(a), c = std::cos(a);
    m4 m; m.c[0] = {c, s, 0, 0}; m.c[1] = {-s, c, 0, 0}; return m;
}

// Right-handed look-at producing a view matrix (world -> eye, -Z forward).
inline m4 lookAt(v3 eye, v3 center, v3 up) {
    v3 f = normalize(center - eye);
    v3 s = normalize(cross(f, up));
    v3 u = cross(s, f);
    m4 m;
    m.c[0] = {s.x, u.x, -f.x, 0};
    m.c[1] = {s.y, u.y, -f.y, 0};
    m.c[2] = {s.z, u.z, -f.z, 0};
    m.c[3] = {-dot(s, eye), -dot(u, eye), dot(f, eye), 1};
    return m;
}

// Right-handed perspective mapping depth to [0,1] (Metal clip convention).
inline m4 perspective(float fovYRad, float aspect, float zn, float zf) {
    float ys = 1.0f / std::tan(fovYRad * 0.5f);
    float xs = ys / aspect;
    float zs = zf / (zn - zf);
    m4 m = m4::zero();
    m.c[0].x = xs;
    m.c[1].y = ys;
    m.c[2].z = zs; m.c[2].w = -1.0f;
    m.c[3].z = zn * zs;
    return m;
}

// Right-handed orthographic mapping depth to [0,1]; used for the shadow cascade.
inline m4 ortho(float l, float r, float b, float t, float zn, float zf) {
    m4 m = m4::zero();
    m.c[0].x = 2.0f / (r - l);
    m.c[1].y = 2.0f / (t - b);
    m.c[2].z = 1.0f / (zn - zf);
    m.c[3] = {(l + r) / (l - r), (b + t) / (b - t), zn / (zn - zf), 1.0f};
    return m;
}

// General inverse via cofactor expansion; used rarely (screen-space picking).
inline m4 inverse(const m4& m) {
    const float* a = &m.c[0].x;
    float inv[16];
    inv[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2]  =  a[1]*a[6]*a[15]  - a[1]*a[7]*a[14]  - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7]  - a[13]*a[3]*a[6];
    inv[6]  = -a[0]*a[6]*a[15]  + a[0]*a[7]*a[14]  + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7]  + a[12]*a[3]*a[6];
    inv[10] =  a[0]*a[5]*a[15]  - a[0]*a[7]*a[13]  - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7]  - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14]  + a[0]*a[6]*a[13]  + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6]  + a[12]*a[2]*a[5];
    inv[3]  = -a[1]*a[6]*a[11]  + a[1]*a[7]*a[10]  + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7]   + a[9]*a[3]*a[6];
    inv[7]  =  a[0]*a[6]*a[11]  - a[0]*a[7]*a[10]  - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7]   - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11]  + a[0]*a[7]*a[9]   + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7]   + a[8]*a[3]*a[5];
    inv[15] =  a[0]*a[5]*a[10]  - a[0]*a[6]*a[9]   - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6]   - a[8]*a[2]*a[5];
    float det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    m4 r;
    if (std::fabs(det) < 1e-12f) return r;
    det = 1.0f / det;
    float* o = &r.c[0].x;
    for (int i = 0; i < 16; i++) o[i] = inv[i] * det;
    return r;
}

// Builds a model matrix for a unit standing on terrain: yaw about Y then translate.
inline m4 trs(v3 pos, float yaw, v3 s) {
    float sn = std::sin(yaw), cs = std::cos(yaw);
    m4 m;
    m.c[0] = {cs * s.x, 0, -sn * s.x, 0};
    m.c[1] = {0, s.y, 0, 0};
    m.c[2] = {sn * s.z, 0, cs * s.z, 0};
    m.c[3] = v4(pos, 1.0f);
    return m;
}

// Angle helpers for turret / facing interpolation.
inline float wrapAngle(float a) {
    while (a > PI) a -= TAU;
    while (a < -PI) a += TAU;
    return a;
}
inline float approachAngle(float cur, float target, float maxStep) {
    float d = wrapAngle(target - cur);
    if (d > maxStep) d = maxStep;
    if (d < -maxStep) d = -maxStep;
    return wrapAngle(cur + d);
}

} // namespace sf
