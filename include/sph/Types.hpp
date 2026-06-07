#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sph {

struct alignas(16) Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float _pad = 0.0f;

    float& operator[](int i) { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};

struct alignas(16) Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

inline Vec3 makeVec3(float x, float y, float z) {
    return Vec3{x, y, z, 0.0f};
}

inline Vec3 operator+(Vec3 a, Vec3 b) { return makeVec3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline Vec3 operator-(Vec3 a, Vec3 b) { return makeVec3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec3 operator*(Vec3 a, float s) { return makeVec3(a.x * s, a.y * s, a.z * s); }
inline Vec3 operator*(float s, Vec3 a) { return a * s; }
inline Vec3 operator/(Vec3 a, float s) { return makeVec3(a.x / s, a.y / s, a.z / s); }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
inline Vec3& operator-=(Vec3& a, Vec3 b) { a = a - b; return a; }
inline Vec3& operator*=(Vec3& a, float s) { a = a * s; return a; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return makeVec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline float lengthSquared(Vec3 a) { return dot(a, a); }
inline float length(Vec3 a) { return std::sqrt(lengthSquared(a)); }
inline Vec3 normalize(Vec3 a) {
    const float l = length(a);
    return l > 0.0f ? a / l : makeVec3(0.0f, 0.0f, 0.0f);
}
inline Vec3 clamp(Vec3 v, Vec3 lo, Vec3 hi) {
    return makeVec3(
        std::clamp(v.x, lo.x, hi.x),
        std::clamp(v.y, lo.y, hi.y),
        std::clamp(v.z, lo.z, hi.z));
}

struct DataSet {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    std::vector<float> densities;
    std::vector<float> internalEnergy;
    std::vector<float> masses;
    std::vector<int32_t> materialIds;
    std::vector<float> pressures;
    std::vector<float> smoothingLengths;
    std::vector<int32_t> particleIds;
    std::vector<float> temperatures;
};

} // namespace sph
