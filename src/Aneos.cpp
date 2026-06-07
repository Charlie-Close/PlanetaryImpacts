#include "sph/Aneos.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "sph/Parameters.hpp"

namespace sph {
namespace {

Vec4 mix(Vec4 a, Vec4 b, float t) {
    return Vec4{
        a.x * (1.0f - t) + b.x * t,
        a.y * (1.0f - t) + b.y * t,
        a.z * (1.0f - t) + b.z * t,
        a.w * (1.0f - t) + b.w * t,
    };
}

Vec4 sampleTable(const AneosTable& table, float u, float v) {
    if (table.data.empty() || table.resolution <= 0) return {};
    const float x = std::clamp(u, 0.0f, 1.0f) * static_cast<float>(table.resolution - 1);
    const float y = std::clamp(v, 0.0f, 1.0f) * static_cast<float>(table.resolution - 1);
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, table.resolution - 1);
    const int y1 = std::min(y0 + 1, table.resolution - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const auto at = [&](int xx, int yy) -> Vec4 { return table.data[static_cast<size_t>(yy) * table.resolution + xx]; };
    return mix(mix(at(x0, y0), at(x1, y0), tx), mix(at(x0, y1), at(x1, y1), tx), ty);
}

float safeLogCoord(float value, float min, float max) {
    value = std::max(value, std::numeric_limits<float>::min());
    return std::log(value / min) / std::log(max / min);
}

} // namespace

AneosTable loadAneosDataFromFile(const std::filesystem::path& path, int resolution) {
    std::ifstream infile(path);
    if (!infile.is_open()) throw std::runtime_error("Could not open ANEOS table: " + path.string());

    std::string line;
    for (int i = 0; i < 13; ++i) std::getline(infile, line);

    int numRho = 0;
    int numT = 0;
    {
        std::getline(infile, line);
        std::istringstream iss(line);
        iss >> numRho >> numT;
    }

    std::vector<float> rhoTemp(numRho);
    for (int count = 0; count < numRho && std::getline(infile, line);) {
        std::istringstream iss(line);
        while (count < numRho && iss >> rhoTemp[count]) ++count;
    }

    std::vector<float> tTemp(numT);
    for (int count = 0; count < numT && std::getline(infile, line);) {
        std::istringstream iss(line);
        while (count < numT && iss >> tTemp[count]) ++count;
    }

    std::vector<float> uTemp(static_cast<size_t>(numRho) * numT);
    std::vector<float> pTemp(static_cast<size_t>(numRho) * numT);
    std::vector<Vec4> dataTemp(static_cast<size_t>(numRho) * numT);
    for (int t = 0, r = 0; t < numT && std::getline(infile, line);) {
        std::istringstream iss(line);
        float u = 0.0f;
        float p = 0.0f;
        float c = 0.0f;
        iss >> u >> p >> c;
        const size_t idx = static_cast<size_t>(r) * numT + t;
        uTemp[idx] = u;
        pTemp[idx] = p;
        dataTemp[idx] = Vec4{p * 1e-18f, c * 1e-6f, tTemp[t], rhoTemp[r]};
        if (++r == numRho) {
            r = 0;
            ++t;
        }
    }

    AneosTable table;
    table.resolution = resolution;
    table.data.resize(static_cast<size_t>(resolution) * resolution);
    std::vector<float> rhoSamples(resolution);
    std::vector<float> uSamples(resolution);
    std::vector<float> pSamples(resolution);
    std::vector<float> tSamples(resolution);

    const float minRho = params::aneosMinRho * 1e6f;
    const float maxRho = params::aneosMaxRho * 1e6f;
    const float minU = params::aneosMinU * 1e12f;
    const float maxU = params::aneosMaxU * 1e12f;
    const float minP = params::aneosMinP * 1e18f;
    const float maxP = params::aneosMaxP * 1e18f;
    for (int i = 0; i < resolution; ++i) {
        const float f = static_cast<float>(i) / static_cast<float>(resolution);
        rhoSamples[i] = minRho * std::pow(maxRho / minRho, f);
        uSamples[i] = minU * std::pow(maxU / minU, f);
        pSamples[i] = minP * std::pow(maxP / minP, f);
        tSamples[i] = params::aneosMinT * std::pow(params::aneosMaxT / params::aneosMinT, f);
    }

    int rhoTi = 0;
    for (int rhoI = 0; rhoI < resolution; ++rhoI) {
        const float rho = rhoSamples[rhoI];
        while (rhoTi < numRho - 2 && rho > rhoTemp[rhoTi + 1]) ++rhoTi;
        const float rhoL = (rho - rhoTemp[rhoTi]) / (rhoTemp[rhoTi + 1] - rhoTemp[rhoTi]);
        const float rhoH = 1.0f - rhoL;
        int ulTi = 0;
        int uhTi = 0;
        for (int uI = 0; uI < resolution; ++uI) {
            const float u = uSamples[uI];
            while (ulTi < numT - 2 && u > uTemp[static_cast<size_t>(rhoTi) * numT + ulTi + 1]) ++ulTi;
            while (uhTi < numT - 2 && u > uTemp[static_cast<size_t>(rhoTi + 1) * numT + uhTi + 1]) ++uhTi;
            const float uLL = (u - uTemp[static_cast<size_t>(rhoTi) * numT + ulTi]) /
                              (uTemp[static_cast<size_t>(rhoTi) * numT + ulTi + 1] - uTemp[static_cast<size_t>(rhoTi) * numT + ulTi]);
            const float uLH = (u - uTemp[static_cast<size_t>(rhoTi + 1) * numT + uhTi]) /
                              (uTemp[static_cast<size_t>(rhoTi + 1) * numT + uhTi + 1] - uTemp[static_cast<size_t>(rhoTi + 1) * numT + uhTi]);
            const Vec4 low = mix(dataTemp[static_cast<size_t>(rhoTi) * numT + ulTi + 1], dataTemp[static_cast<size_t>(rhoTi) * numT + ulTi], uLL);
            const Vec4 high = mix(dataTemp[static_cast<size_t>(rhoTi + 1) * numT + uhTi + 1], dataTemp[static_cast<size_t>(rhoTi + 1) * numT + uhTi], uLH);
            table.data[static_cast<size_t>(rhoI) * resolution + uI] = mix(high, low, rhoL);
            (void)rhoH;
        }
    }

    int tRhoi = 0;
    for (int tI = 0; tI < resolution; ++tI) {
        const float temp = tSamples[tI];
        while (tRhoi < numT - 2 && temp > tTemp[tRhoi + 1]) ++tRhoi;
        const float tL = (temp - tTemp[tRhoi]) / (tTemp[tRhoi + 1] - tTemp[tRhoi]);
        const float tH = 1.0f - tL;
        int plRhoi = 0;
        int phRhoi = 0;
        for (int pI = 0; pI < resolution; ++pI) {
            const float p = pSamples[pI];
            while (plRhoi < numRho - 2 && p > pTemp[static_cast<size_t>(plRhoi + 1) * numT + tRhoi]) ++plRhoi;
            while (phRhoi < numRho - 2 && p > pTemp[static_cast<size_t>(phRhoi + 1) * numT + tRhoi + 1]) ++phRhoi;
            const float pLL = (p - pTemp[static_cast<size_t>(plRhoi) * numT + tRhoi]) /
                              std::max(pTemp[static_cast<size_t>(plRhoi + 1) * numT + tRhoi] - pTemp[static_cast<size_t>(plRhoi) * numT + tRhoi], 1e-24f);
            const float pLH = (p - pTemp[static_cast<size_t>(phRhoi) * numT + tRhoi + 1]) /
                              std::max(pTemp[static_cast<size_t>(phRhoi + 1) * numT + tRhoi + 1] - pTemp[static_cast<size_t>(phRhoi) * numT + tRhoi + 1], 1e-24f);
            float rho = tL * (rhoTemp[plRhoi] * pLL + rhoTemp[plRhoi + 1] * (1.0f - pLL)) +
                        tH * (rhoTemp[phRhoi] * pLH + rhoTemp[phRhoi + 1] * (1.0f - pLH));
            if (rho < 1e-24f || rho > 1e24f || !std::isfinite(rho)) rho = rhoTemp[phRhoi + 1];
            table.data[static_cast<size_t>(tI) * resolution + pI].w = rho;
        }
    }
    return table;
}

AneosTable loadHmDataFromFile(const std::filesystem::path& path) {
    std::ifstream infile(path);
    if (!infile.is_open()) throw std::runtime_error("Could not open HM table: " + path.string());
    std::string line;
    for (int i = 0; i < 12; ++i) std::getline(infile, line);

    int numRho = 0;
    int numU = 0;
    float logRhoMin = 0.0f;
    float logRhoMax = 0.0f;
    float logUMin = 0.0f;
    float logUMax = 0.0f;
    std::getline(infile, line);
    std::istringstream(line) >> logRhoMin >> logRhoMax >> numRho >> logUMin >> logUMax >> numU;
    (void)logUMin;
    (void)logUMax;

    AneosTable table;
    table.resolution = numRho;
    std::vector<float> p(static_cast<size_t>(numRho) * numU);
    std::vector<float> t(static_cast<size_t>(numRho) * numU);
    for (size_t count = 0; count < p.size() && std::getline(infile, line);) {
        std::istringstream iss(line);
        while (count < p.size() && iss >> p[count]) ++count;
    }
    for (size_t count = 0; count < t.size() && std::getline(infile, line);) {
        std::istringstream iss(line);
        while (count < t.size() && iss >> t[count]) ++count;
    }

    std::vector<float> rho(numRho);
    for (int i = 0; i < numRho; ++i) {
        rho[i] = std::exp(logRhoMin + static_cast<float>(i) * (logRhoMax - logRhoMin) / static_cast<float>(numRho - 1));
    }

    table.data.resize(p.size());
    for (int j = 0; j < numU; ++j) {
        for (int i = 0; i < numRho; ++i) {
            const size_t idx = static_cast<size_t>(i) * numU + j;
            const float dpDrho = std::max(1.3f * p[idx] / rho[i], 0.0f);
            table.data[idx] = Vec4{p[idx] * 1e-18f, std::sqrt(dpDrho) * 1e-6f, t[idx], 0.0f};
        }
    }
    return table;
}

EquationOfState EquationOfState::load(const std::filesystem::path& root) {
    EquationOfState eos;
    eos.iron = loadAneosDataFromFile(root / "ANEOS_iron_S20.txt", params::aneosTextureResolution);
    eos.forsterite = loadAneosDataFromFile(root / "ANEOS_forsterite_S19.txt", params::aneosTextureResolution);
    eos.fe85si15 = loadAneosDataFromFile(root / "ANEOS_Fe85Si15_S20.txt", params::aneosTextureResolution);
    eos.hhe = loadHmDataFromFile(root / "HM80_HHe.txt");
    eos.ice = loadHmDataFromFile(root / "HM80_ice.txt");
    eos.rock = loadHmDataFromFile(root / "HM80_rock.txt");
    return eos;
}

Vec4 EquationOfState::sample(int materialId, float internalEnergy, float density) const {
    switch (materialId) {
    case 200:
        return sampleTable(hhe, safeLogCoord(internalEnergy, 1e-8f, 5e9f), safeLogCoord(density, 1e-7f, 1000.0f));
    case 201:
        return sampleTable(ice, safeLogCoord(internalEnergy, 1e-9f, 5e9f), safeLogCoord(density, 1e-6f, 6000.0f));
    case 202:
        return sampleTable(rock, safeLogCoord(internalEnergy, 1e-8f, 1e9f), safeLogCoord(density, 1e-6f, 20000.0f));
    case 400:
        return sampleTable(forsterite, safeLogCoord(internalEnergy, params::aneosMinU, params::aneosMaxU),
                           safeLogCoord(density, params::aneosMinRho, params::aneosMaxRho));
    case 401:
        return sampleTable(iron, safeLogCoord(internalEnergy, params::aneosMinU, params::aneosMaxU),
                           safeLogCoord(density, params::aneosMinRho, params::aneosMaxRho));
    case 402:
        return sampleTable(fe85si15, safeLogCoord(internalEnergy, params::aneosMinU, params::aneosMaxU),
                           safeLogCoord(density, params::aneosMinRho, params::aneosMaxRho));
    default:
        return Vec4{0.0f, 1.0f, 0.0f, density};
    }
}

float EquationOfState::apparentDensity(int materialId, float temperature, float pressure) const {
    const float tCoord = safeLogCoord(temperature, params::aneosMinT, params::aneosMaxT);
    const float pCoord = safeLogCoord(pressure, params::aneosMinP, params::aneosMaxP);
    switch (materialId) {
    case 400: return sampleTable(forsterite, pCoord, tCoord).w;
    case 401: return sampleTable(iron, pCoord, tCoord).w;
    case 402: return sampleTable(fe85si15, pCoord, tCoord).w;
    default: return 1.0f;
    }
}

float EquationOfState::densityFactor(int materialA, int materialB, float temperature, float pressure) const {
    if (materialA < 400 || materialB < 400 || materialA == materialB) return 1.0f;
    const float b = apparentDensity(materialB, temperature, pressure);
    return b > 0.0f ? apparentDensity(materialA, temperature, pressure) / b : 1.0f;
}

} // namespace sph
