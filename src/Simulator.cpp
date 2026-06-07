#include "sph/Simulator.hpp"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include "sph/Hdf5.hpp"
#include "sph/Parameters.hpp"

namespace sph {
namespace {

uint32_t splitBy3(uint32_t a) {
    uint32_t x = a & 0x000003ffu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8)) & 0x0300F00Fu;
    x = (x | (x << 4)) & 0x030C30C3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
}

float WFast(float r, float H1, float fac) {
    const float u = r * H1;
    if (u > 1.0f) return 0.0f;
    const float u2 = u * u;
    const float u3 = u2 * u;
    if (u > 0.5f) return fac * (-u3 + 3.0f * u2 - 3.0f * u + 1.0f);
    return fac * (3.0f * u3 - 3.0f * u2 + 0.5f);
}

Vec3 gradWFast(Vec3 xij, float r, float r1, float H1, float fac) {
    const float u = r * H1;
    if (u > 1.0f) return {};
    const float u2 = u * u;
    const Vec3 rHat = xij * r1;
    if (u > 0.5f) return rHat * (fac * (-3.0f * u2 + 6.0f * u - 3.0f));
    return rHat * (fac * (9.0f * u2 - 6.0f * u));
}

float dWdhFast(float r, float H1, float fac) {
    const float u = r * H1;
    if (u > 1.0f) return 0.0f;
    const float u2 = u * u;
    const float u3 = u2 * u;
    if (u > 0.5f) return -fac * (-6.0f * u3 + 15.0f * u2 - 12.0f * u + 3.0f);
    return -fac * (18.0f * u3 - 15.0f * u2 + 1.5f);
}

float dphiDr(Vec3 xij, float eta) {
    const float eta1 = 1.0f / eta;
    const float r = length(xij);
    const float q = r * eta1;
    if (q <= 1.0f) return eta1 * eta1 * ((4.0f / 3.0f) * q - (6.0f / 5.0f) * q * q * q + 0.5f * q * q * q * q);
    if (q <= 2.0f) {
        const float q1 = 1.0f / q;
        return eta1 * eta1 * ((8.0f / 3.0f) * q - 3.0f * q * q + (6.0f / 5.0f) * q * q * q -
                              (1.0f / 6.0f) * q * q * q * q - (1.0f / 15.0f) * q1 * q1);
    }
    const float r1 = 1.0f / r;
    return r1 * r1;
}

} // namespace

float Simulator::paramsMinDt() { return params::minDt; }

Simulator::Simulator(DataSet data, EquationOfState eos, int maxParticles)
    : data_(std::move(data)), eos_(std::move(eos)) {
    if (maxParticles > 0 && static_cast<int>(data_.positions.size()) > maxParticles) {
        const size_t n = static_cast<size_t>(maxParticles);
        data_.positions.resize(n);
        data_.velocities.resize(n);
        data_.densities.resize(n);
        data_.internalEnergy.resize(n);
        data_.masses.resize(n);
        data_.materialIds.resize(n);
        data_.pressures.resize(n);
        data_.smoothingLengths.resize(n);
        data_.particleIds.resize(n);
        data_.temperatures.resize(n);
    }
}

void Simulator::initialise() {
    const size_t n = data_.positions.size();
    active_.assign(n, true);
    alive_.assign(n, true);
    accelerations_.assign(n, {});
    accelerations1_.assign(n, {});
    gravAccelerations_.assign(n, {});
    rhoGrads_.assign(n, {});
    gradientTerms_.assign(n, 1.0f);
    speedOfSound_.assign(n, 1.0f);
    dInternalEnergy_.assign(n, 0.0f);
    dInternalEnergy1_.assign(n, 0.0f);
    balsara_.assign(n, 0.0f);
    dhDt_.assign(n, 0.0f);
    alpha_.assign(n, params::viscosityAlpha);
    daDt_.assign(n, 0.0f);
    alphaLoc_.assign(n, params::viscosityAlpha);
    pAlphaLoc_.assign(n, params::viscosityAlpha);
    localMaxH_.assign(n, params::maxSmoothingLength);
    nextActiveTime_.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        Vec3 translated = data_.positions[i] - makeVec3(params::boxCenter, params::boxCenter, params::boxCenter);
        alive_[i] = std::abs(translated.x) <= params::boxSize && std::abs(translated.y) <= params::boxSize && std::abs(translated.z) <= params::boxSize;
    }
    sortParticles();
}

uint32_t Simulator::cellIndexFromCell(int x, int y, int z) const {
    const int mask = (1 << params::cellPower) - 1;
    return morton3D(static_cast<uint32_t>(x & mask), static_cast<uint32_t>(y & mask), static_cast<uint32_t>(z & mask));
}

uint32_t Simulator::cellIndexFromPosition(Vec3 pos) const {
    return cellIndexFromCell(static_cast<int>(std::floor(pos.x / params::cellWidth)),
                             static_cast<int>(std::floor(pos.y / params::cellWidth)),
                             static_cast<int>(std::floor(pos.z / params::cellWidth)));
}

void Simulator::sortParticles() {
    const size_t n = data_.positions.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return cellIndexFromPosition(data_.positions[a]) < cellIndexFromPosition(data_.positions[b]);
    });
    auto reorder = [&](auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        std::vector<T> tmp(values.size());
        for (size_t i = 0; i < order.size(); ++i) tmp[i] = values[order[i]];
        values.swap(tmp);
    };
    reorder(data_.positions);
    reorder(data_.velocities);
    reorder(data_.densities);
    reorder(data_.internalEnergy);
    reorder(data_.masses);
    reorder(data_.materialIds);
    reorder(data_.pressures);
    reorder(data_.smoothingLengths);
    reorder(data_.particleIds);
    reorder(data_.temperatures);
    reorder(active_);
    reorder(alive_);
    reorder(accelerations_);
    reorder(accelerations1_);
    reorder(gravAccelerations_);
    reorder(rhoGrads_);
    reorder(gradientTerms_);
    reorder(speedOfSound_);
    reorder(dInternalEnergy_);
    reorder(dInternalEnergy1_);
    reorder(balsara_);
    reorder(dhDt_);
    reorder(alpha_);
    reorder(daDt_);
    reorder(alphaLoc_);
    reorder(pAlphaLoc_);
    reorder(localMaxH_);
    reorder(nextActiveTime_);
    rebuildCellRanges();
}

void Simulator::rebuildCellRanges() {
    cellData_.resize(data_.positions.size());
    for (uint32_t i = 0; i < data_.positions.size(); ++i) {
        cellData_[i] = {cellIndexFromPosition(data_.positions[i]), i};
    }
    cellRanges_.clear();
    for (uint32_t i = 0; i < cellData_.size();) {
        const uint32_t cell = cellData_[i].first;
        uint32_t j = i + 1;
        while (j < cellData_.size() && cellData_[j].first == cell) ++j;
        cellRanges_[cell] = CellRange{i, j};
        i = j;
    }
}

std::vector<uint32_t> Simulator::cellsToScan(Vec3 position, float h) const {
    const float radius = params::gamma * h;
    const Vec3 minRange = position - makeVec3(radius, radius, radius);
    const Vec3 maxRange = position + makeVec3(radius, radius, radius);
    const int minX = static_cast<int>(std::floor(minRange.x / params::cellWidth));
    const int minY = static_cast<int>(std::floor(minRange.y / params::cellWidth));
    const int minZ = static_cast<int>(std::floor(minRange.z / params::cellWidth));
    const int maxX = static_cast<int>(std::floor(maxRange.x / params::cellWidth));
    const int maxY = static_cast<int>(std::floor(maxRange.y / params::cellWidth));
    const int maxZ = static_cast<int>(std::floor(maxRange.z / params::cellWidth));
    std::vector<uint32_t> cells;
    cells.reserve(static_cast<size_t>((maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1)));
    for (int x = minX; x <= maxX; ++x)
        for (int y = minY; y <= maxY; ++y)
            for (int z = minZ; z <= maxZ; ++z)
                cells.push_back(cellIndexFromCell(x, y, z));
    return cells;
}

Vec4 Simulator::eos(int materialId, float internalEnergy, float density) const {
    return eos_.sample(materialId, internalEnergy, density);
}

void Simulator::activatePass() {
    for (size_t i = 0; i < active_.size(); ++i) active_[i] = globalTime_ >= nextActiveTime_[i];
}

void Simulator::gravityPass() {
    const size_t n = data_.positions.size();
    std::fill(gravAccelerations_.begin(), gravAccelerations_.end(), Vec3{});
    if (n > 20000) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!alive_[i]) continue;
        Vec3 acc{};
        for (size_t j = 0; j < n; ++j) {
            if (i == j || !alive_[j]) continue;
            Vec3 xij = data_.positions[i] - data_.positions[j];
            const float r = length(xij);
            if (r == 0.0f) continue;
            const float eta = std::min(data_.smoothingLengths[j] * params::gamma * params::plumberEquivalent, params::gravitySmoothingLength);
            acc -= normalize(xij) * (data_.masses[j] * dphiDr(xij, eta));
        }
        gravAccelerations_[i] = acc * params::gravityG;
    }
}

void Simulator::densityPass() {
    if (dt_ == 0) dt_ = static_cast<int>(std::floor(params::maxDt / params::minDt));
    for (size_t i = 0; i < data_.positions.size(); ++i) {
        if (!alive_[i]) continue;
        if (!active_[i]) {
            Vec4 pc = eos(data_.materialIds[i], data_.internalEnergy[i], data_.densities[i]);
            data_.pressures[i] = pc.x;
            speedOfSound_[i] = pc.y;
            data_.temperatures[i] = pc.z;
            continue;
        }
        float h = data_.smoothingLengths[i];
        float density = 0.0f;
        float densityH = 0.0f;
        int particlesInRange = 0;
        float minBound = 0.0f;
        float maxBound = params::maxSmoothingLength;
        float omega = 1.0f;
        for (int iter = 0; iter < params::maxDensityIterations; ++iter) {
            density = 0.0f;
            densityH = 0.0f;
            particlesInRange = 0;
            const float H1 = params::gamma1 / h;
            const float kernelFac = params::kernelConstant * H1 * H1 * H1;
            const float dhKernelFac = kernelFac * H1;
            const float support = params::gamma * h;
            const float support2 = support * support;
            for (uint32_t cell : cellsToScan(data_.positions[i], h)) {
                auto found = cellRanges_.find(cell);
                if (found == cellRanges_.end()) continue;
                for (uint32_t k = found->second.start; k < found->second.end; ++k) {
                    const uint32_t j = cellData_[k].second;
                    if (!alive_[j]) continue;
                    const float r2 = lengthSquared(data_.positions[j] - data_.positions[i]);
                    if (r2 > support2) continue;
                    const float r = std::sqrt(r2);
                    const float densityFactor = eos_.densityFactor(data_.materialIds[i], data_.materialIds[j], data_.temperatures[j], data_.pressures[j]);
                    const float mass = densityFactor * data_.masses[j];
                    density += WFast(r, H1, kernelFac) * mass;
                    densityH += mass * dWdhFast(r, H1, dhKernelFac);
                    ++particlesInRange;
                }
            }
            if (density <= 0.0f) break;
            omega = 1.0f + densityH * h / (3.0f * density);
            const float etaH = params::resolutionEta / h;
            const float eta = data_.masses[i] * etaH * etaH * etaH - density;
            float newH = std::clamp(h * (1.0f + eta / (3.0f * density * omega)), 1e-12f, params::maxSmoothingLength);
            if (particlesInRange < params::particleInRangeThreshold) newH = std::min(std::max(newH, h * 2.0f), params::maxSmoothingLength);
            if (newH < h) maxBound = h; else minBound = h;
            if (newH < minBound || newH > maxBound) newH = 0.5f * (minBound + maxBound);
            const float eps = std::abs(newH - h) / h;
            h = newH;
            if (eps <= params::densityEta) break;
        }
        Vec3 rhoGrad{};
        Vec3 velCurl{};
        float velDiv = 0.0f;
        float unscaledDensity = 0.0f;
        float localMax = h;
        const float H1 = params::gamma1 / h;
        const float kernelFac = params::kernelConstant * H1 * H1 * H1;
        const float gradFac = kernelFac * H1;
        const float support = params::gamma * h;
        const float support2 = support * support;
        for (uint32_t cell : cellsToScan(data_.positions[i], h)) {
            auto found = cellRanges_.find(cell);
            if (found == cellRanges_.end()) continue;
            for (uint32_t k = found->second.start; k < found->second.end; ++k) {
                const uint32_t j = cellData_[k].second;
                if (!alive_[j]) continue;
                Vec3 xij = data_.positions[j] - data_.positions[i];
                const float r2 = lengthSquared(xij);
                if (r2 > support2) continue;
                const float r = std::sqrt(r2);
                const float Wij = WFast(r, H1, kernelFac);
                unscaledDensity += data_.masses[j] * Wij;
                if (r <= 1e-12f) continue;
                const Vec3 vij = data_.velocities[j] - data_.velocities[i];
                const Vec3 gW = gradWFast(xij, r, 1.0f / r, H1, gradFac);
                velDiv += data_.masses[j] * dot(vij, gW);
                velCurl += cross(vij, gW) * data_.masses[j];
                rhoGrad += gW * (data_.masses[j] * density);
                localMax = std::max(localMax, data_.smoothingLengths[j]);
            }
        }
        localMaxH_[i] = localMax;
        data_.smoothingLengths[i] = h;
        data_.densities[i] = density;
        rhoGrads_[i] = density > 0.0f ? rhoGrad / density : Vec3{};
        Vec4 pc = eos(data_.materialIds[i], data_.internalEnergy[i], density);
        data_.pressures[i] = pc.x;
        speedOfSound_[i] = pc.y;
        data_.temperatures[i] = pc.z;
        const float absCurl = unscaledDensity > 0.0f ? length(velCurl) / unscaledDensity : 0.0f;
        velDiv = unscaledDensity > 0.0f ? velDiv / unscaledDensity : 0.0f;
        balsara_[i] = std::abs(velDiv) / (std::abs(velDiv) + absCurl + 1e-4f * (pc.y / h));
        gradientTerms_[i] = omega != 0.0f ? 1.0f / omega : 1.0f;
        pAlphaLoc_[i] = params::alphaMin;
    }
}

void Simulator::accelerationPass() {
    dt_ = static_cast<int>(std::floor(params::maxDt / params::minDt));
    std::fill(accelerations_.begin(), accelerations_.end(), Vec3{});
    for (size_t i = 0; i < data_.positions.size(); ++i) {
        if (!alive_[i]) continue;
        if (!active_[i]) {
            dt_ = std::min(dt_, std::max(nextActiveTime_[i] - globalTime_, 1));
            continue;
        }
        const float rhoI = std::max(data_.densities[i], 1e-20f);
        const float factI = data_.pressures[i] * gradientTerms_[i] / (rhoI * rhoI);
        const float hi = data_.smoothingLengths[i];
        const float H1 = params::gamma1 / hi;
        const float gradFac = params::kernelConstant * H1 * H1 * H1 * H1;
        Vec3 dv{};
        float du = 0.0f;
        float dh = 0.0f;
        float vSigI = 2.0f * speedOfSound_[i];
        for (uint32_t cell : cellsToScan(data_.positions[i], localMaxH_[i])) {
            auto found = cellRanges_.find(cell);
            if (found == cellRanges_.end()) continue;
            for (uint32_t k = found->second.start; k < found->second.end; ++k) {
                const uint32_t j = cellData_[k].second;
                if (i == j || !alive_[j]) continue;
                Vec3 xij = data_.positions[i] - data_.positions[j];
                const float r2 = lengthSquared(xij);
                const float hj = data_.smoothingLengths[j];
                if ((r2 > params::gamma * params::gamma * hi * hi && r2 > params::gamma * params::gamma * hj * hj) || r2 < 1e-12f) continue;
                const float r = std::sqrt(r2);
                const float r1 = 1.0f / r;
                const float rhoJ = std::max(data_.densities[j], 1e-20f);
                const float factJ = data_.pressures[j] * gradientTerms_[j] / (rhoJ * rhoJ);
                const Vec3 vij = data_.velocities[i] - data_.velocities[j];
                const float mu = dot(vij, xij) < 0.0f ? dot(vij, xij) * r1 : 0.0f;
                const float vSig = speedOfSound_[i] + speedOfSound_[j] - params::viscosityBeta * mu;
                vSigI = std::max(vSigI, vSig);
                const float nu = -0.25f * (alpha_[i] + alpha_[j]) * (balsara_[i] + balsara_[j]) * mu * vSig / (rhoI + rhoJ);
                const Vec3 gWi = gradWFast(xij, r, r1, H1, gradFac);
                const float Hj1 = params::gamma1 / hj;
                const Vec3 gWj = gradWFast(xij, r, r1, Hj1, params::kernelConstant * Hj1 * Hj1 * Hj1 * Hj1);
                const Vec3 visc = (gWi + gWj) * (0.5f * nu);
                const Vec3 sphAcc = gWi * factI + gWj * factJ;
                dv -= (sphAcc + visc) * data_.masses[j];
                du += data_.masses[j] * (factI * dot(vij, gWi) + 0.5f * dot(vij, visc));
                dh += (data_.masses[j] / rhoJ) * dot(vij, gWi);
            }
        }
        const float goalDt = std::clamp(2.0f * params::cfl * params::gamma * hi / std::max(vSigI, 1e-12f), params::minDt, params::maxDt);
        int integerDt = 1;
        while (2.0f * params::minDt * integerDt < goalDt && 2.0f * params::minDt * integerDt < params::maxDt) integerDt += integerDt;
        nextActiveTime_[i] = globalTime_ + integerDt;
        dt_ = std::min(dt_, integerDt);
        accelerations_[i] = dv;
        dInternalEnergy_[i] = du;
        dhDt_[i] = -0.333333333f * hi * dh;
    }
}

void Simulator::accelerationStepPass() {
    accelerations1_ = accelerations_;
    dInternalEnergy1_ = dInternalEnergy_;
}

void Simulator::integratePass() {
    globalTime_ += dt_;
    const float dt = params::minDt * static_cast<float>(dt_);
    for (size_t i = 0; i < data_.positions.size(); ++i) {
        if (!alive_[i]) continue;
        if (active_[i]) {
            accelerations_[i] = accelerations1_[i];
            dInternalEnergy_[i] = dInternalEnergy1_[i];
        }
        const Vec3 a = accelerations_[i] + gravAccelerations_[i];
        data_.positions[i] += data_.velocities[i] * dt + a * (0.5f * dt * dt);
        data_.velocities[i] += a * dt;
        const float oldH = data_.smoothingLengths[i];
        data_.smoothingLengths[i] = oldH * std::exp((1.0f / oldH) * dhDt_[i] * dt);
        data_.densities[i] *= std::exp(-(3.0f / oldH) * dhDt_[i] * dt);
        dhDt_[i] *= data_.smoothingLengths[i] / oldH;
        data_.internalEnergy[i] = std::max(data_.internalEnergy[i] + dt * dInternalEnergy_[i], 0.0f);
        Vec3 translated = data_.positions[i] - makeVec3(params::boxCenter, params::boxCenter, params::boxCenter);
        alive_[i] = std::abs(translated.x) <= params::boxSize && std::abs(translated.y) <= params::boxSize && std::abs(translated.z) <= params::boxSize;
    }
}

void Simulator::step() {
    throw std::runtime_error("CPU simulation stepping has been removed; simulation must run through the Vulkan GPU path.");
}

void Simulator::run(const RunOptions&) {
    throw std::runtime_error("CPU simulation run has been removed; simulation must run through the Vulkan GPU path.");
}

void Simulator::saveState(const std::filesystem::path& path) const {
    std::filesystem::create_directories(path.parent_path().empty() ? "." : path.parent_path());
    writeHdf5(path, data_);
}

void Simulator::writeSnapshot(const std::filesystem::path& path, Vec3 cameraPosition, float pitch, float yaw) const {
    (void)path;
    (void)cameraPosition;
    (void)pitch;
    (void)yaw;
    throw std::runtime_error("CPU snapshot rendering is disabled; use VulkanSimulation::writeSnapshot for GPU-only Metal-equivalent snapshots");
}

} // namespace sph
