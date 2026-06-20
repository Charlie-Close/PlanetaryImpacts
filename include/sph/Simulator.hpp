#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include "Aneos.hpp"
#include "Types.hpp"

namespace sph {

struct RunOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path snapshotDir;
    int steps = 1;
    int maxParticles = 0;
    int videoIntervalSeconds = 1000;
    bool snapshots = true;
    bool saveState = true;
    bool validateOnly = false;
    bool requireVulkan = true;
    bool headless = false;
    int windowWidth = 1024;
    int windowHeight = 1024;
    int viewerFrames = 0;
};

class Simulator {
public:
    explicit Simulator(DataSet data, EquationOfState eos, int maxParticles = 0);

    int particleCount() const { return static_cast<int>(data_.positions.size()); }
    float time() const { return static_cast<float>(globalTime_) * paramsMinDt(); }
    const DataSet& data() const { return data_; }
    DataSet& mutableData() { return data_; }
    const EquationOfState& equationOfState() const { return eos_; }
    void setGlobalTimeTicks(int32_t ticks) { globalTime_ = ticks; }

    void initialise();
    void step();
    void run(const RunOptions& options);
    void saveState(const std::filesystem::path& path) const;
    void writeSnapshot(const std::filesystem::path& path, Vec3 cameraPosition, float pitch, float yaw) const;

private:
    struct CellRange {
        uint32_t start = 0;
        uint32_t end = 0;
    };

    static float paramsMinDt();
    void sortParticles();
    void activatePass();
    void gravityPass();
    void densityPass();
    void accelerationPass();
    void accelerationStepPass();
    void integratePass();
    void rebuildCellRanges();

    std::vector<uint32_t> cellsToScan(Vec3 position, float h) const;
    uint32_t cellIndexFromPosition(Vec3 pos) const;
    uint32_t cellIndexFromCell(int x, int y, int z) const;
    Vec4 eos(int materialId, float internalEnergy, float density) const;

    DataSet data_;
    EquationOfState eos_;
    std::vector<bool> active_;
    std::vector<bool> alive_;
    std::vector<Vec3> accelerations_;
    std::vector<Vec3> accelerations1_;
    std::vector<Vec3> gravAccelerations_;
    std::vector<Vec3> rhoGrads_;
    std::vector<float> gradientTerms_;
    std::vector<float> speedOfSound_;
    std::vector<float> dInternalEnergy_;
    std::vector<float> dInternalEnergy1_;
    std::vector<float> balsara_;
    std::vector<float> dhDt_;
    std::vector<float> alpha_;
    std::vector<float> daDt_;
    std::vector<float> alphaLoc_;
    std::vector<float> pAlphaLoc_;
    std::vector<float> localMaxH_;
    std::vector<int32_t> nextActiveTime_;
    std::vector<std::pair<uint32_t, uint32_t>> cellData_;
    std::unordered_map<uint32_t, CellRange> cellRanges_;
    int32_t globalTime_ = 0;
    int32_t dt_ = 1;
};

} // namespace sph
