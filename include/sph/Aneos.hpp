#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include "Types.hpp"

namespace sph {

struct AneosTable {
    int resolution = 0;
    std::vector<Vec4> data;
};

struct EquationOfState {
    AneosTable iron;
    AneosTable forsterite;
    AneosTable fe85si15;
    AneosTable hhe;
    AneosTable ice;
    AneosTable rock;

    static EquationOfState load(const std::filesystem::path& root);
    Vec4 sample(int materialId, float internalEnergy, float density) const;
    float apparentDensity(int materialId, float temperature, float pressure) const;
    float densityFactor(int materialA, int materialB, float temperature, float pressure) const;
};

AneosTable loadAneosDataFromFile(const std::filesystem::path& path, int resolution);
AneosTable loadHmDataFromFile(const std::filesystem::path& path);

} // namespace sph

