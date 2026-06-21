#pragma once

#include <array>
#include "Types.hpp"

namespace sph::params {

inline constexpr float viscosityAlpha = 1.5f;
inline constexpr float viscosityBeta = 3.0f;
inline constexpr float alphaMin = 0.1f;
inline constexpr float alphaMax = 2.0f;
inline constexpr float densityEta = 0.001f;
inline constexpr float maxSmoothingLength = 1.0f;
inline constexpr int maxDensityIterations = 50;
inline constexpr int neighboursEstimate = 128;
inline constexpr float resolutionEta = 1.2348f;
inline constexpr int particleInRangeThreshold = 10;

inline constexpr int aneosTextureResolution = 2048;
inline constexpr float aneosMinRho = 1e-6f;
inline constexpr float aneosMaxRho = 1.0f;
inline constexpr float aneosMinU = 1e-10f;
inline constexpr float aneosMaxU = 1.0f;
inline constexpr float aneosMinP = 1e-7f;
inline constexpr float aneosMaxP = 1e-3f;
inline constexpr float aneosMinT = 5e2f;
inline constexpr float aneosMaxT = 1e5f;

inline constexpr float cfl = 0.2f;
inline constexpr float gdt = 0.5f;
inline constexpr float minDt = 0.001f;
inline constexpr float maxDt = 100.0f;
inline constexpr int densityGradientSettlingIterations = 3;
inline constexpr int activateAllSteps = 1024;

inline constexpr float gamma = 1.825742f;
inline constexpr float gamma1 = 1.0f / gamma;
inline constexpr float kernelConstant = 16.0f / 3.14159265358979323846f;

inline constexpr float gravityG = 6.67e-5f;
inline constexpr float gravitySmoothingLength = 0.12f;
inline constexpr float plumberEquivalent = 3.0f;
inline constexpr float gravityEta = 0.001f;
inline constexpr int multipoleExpansionPower = 4;
static_assert(multipoleExpansionPower >= 1 && multipoleExpansionPower <= 4,
              "Vulkan multipole expansion currently supports powers 1 through 4");
inline constexpr int expansionTerms =
    ((multipoleExpansionPower + 1) * (multipoleExpansionPower + 2) * (multipoleExpansionPower + 3)) / 6;
inline constexpr int multipolePowerTerms = multipoleExpansionPower + 1;
inline constexpr int maxUncheckedPointers = 32;
inline constexpr int maxLocalUncheckedPointers = 256;
inline constexpr int gravityMaxRecursion = 8;
inline constexpr float maxFloat = 3.402823466e38f;

inline constexpr float cellWidth = 0.30f;
inline constexpr int cellPower = 7;
inline constexpr int sortingBlockSize = 256;
inline constexpr int sortingMaskLength = 8;
inline constexpr int sortingBucketNumber = 1 << sortingMaskLength;
inline constexpr int sortingBitMask = sortingBucketNumber - 1;
inline constexpr int shuffleFrames = 64;

inline constexpr float particleSize = 160.0f;
inline constexpr int snapshotResolution = 1024;
inline constexpr int startSnapshot = 0;
inline constexpr int snapshotPeriodSeconds = 25;
inline constexpr int nSnapshotters = 4;
inline constexpr int boxCenter = 318;
inline constexpr int boxSize = 318;
inline constexpr Vec3 startingPosition = {315.0f, 355.0f, 100.0f, 0.0f};
inline constexpr float startingPitch = -10.0f;
inline constexpr float startingYaw = 90.0f;
inline constexpr Vec3 lightDirection = {-1.0f, -1.0f, 0.5f, 0.0f};
inline constexpr std::array<Vec3, 4> snapshotPositions = {
    Vec3{315.0f, 355.0f, 100.0f, 0.0f},
    Vec3{350.0f, 355.0f, 260.0f, 0.0f},
    Vec3{262.0f, 362.0f, 280.0f, 0.0f},
    Vec3{248.0f, 306.0f, 238.0f, 0.0f},
};
inline constexpr std::array<float, 4> snapshotPitches = {-10.0f, -30.0f, -34.0f, 3.0f};
inline constexpr std::array<float, 4> snapshotYaws = {90.0f, 120.0f, 40.0f, 53.0f};

inline constexpr const char* defaultInput = "demo_impact_n50.hdf5";
inline constexpr const char* fallbackInput = "saves_n70_cold/state_33.hdf5";
inline constexpr const char* defaultSnapshotDir = "snapshots_n70_cold";
inline constexpr const char* defaultSavesDir = "saves_n70_cold";

} // namespace sph::params
