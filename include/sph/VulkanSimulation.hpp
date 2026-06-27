#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <future>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include "Octree.hpp"
#include "Parameters.hpp"
#include "Simulator.hpp"

namespace sph {

class VulkanSimulation {
public:
    explicit VulkanSimulation(Simulator& simulator);
    VulkanSimulation(Simulator& simulator,
                     VkPhysicalDevice physicalDevice,
                     VkDevice device,
                     VkQueue queue,
                     uint32_t queueFamily,
                     std::string deviceName);
    ~VulkanSimulation();

    VulkanSimulation(const VulkanSimulation&) = delete;
    VulkanSimulation& operator=(const VulkanSimulation&) = delete;

    void step(int maxTicks = 0);
    void syncToSimulator();
    void writeSnapshot(const std::filesystem::path& path, Vec3 cameraPosition, float pitch, float yaw);
    int globalTimeTicks() const { return globalTimeTicks_; }
    const std::string& deviceName() const { return deviceName_; }
    uint32_t particleCount() const { return particleCount_; }

    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        void* mapped = nullptr;
    };
    struct ChunkedBuffer {
        std::vector<Buffer> chunks;
        VkDeviceSize logicalSize = 0;
        VkDeviceSize elementSize = 0;
        uint32_t elementsPerChunk = 0;
    };
    const Buffer& positionsBuffer() const { return positions_; }
    const Buffer& densitiesBuffer() const { return densities_; }
    const Buffer& smoothingLengthsBuffer() const { return smoothingLengths_; }
    const Buffer& temperaturesBuffer() const { return temperatures_; }
    const Buffer& materialIdsBuffer() const { return materialIds_; }
    const Buffer& rhoGradsBuffer() const { return rhoGrads_; }
    const Buffer& alphaBuffer() const { return alpha_; }

private:
    static constexpr int ExpansionTerms = params::expansionTerms;
    static constexpr int MultipolePowerTerms = params::multipolePowerTerms;
    static constexpr int MaxUncheckedPointers = params::maxUncheckedPointers;

    struct alignas(16) GpuMultipole {
        Vec4 pos{};
        Vec4 min{};
        Vec4 max{};
        float size = 0.0f;
        float expansion[ExpansionTerms]{};
        float power[MultipolePowerTerms]{};
        float minGrav = 0.0f;
        float eta = 0.0f;
        float _pad[2]{};
    };

    struct alignas(16) GpuLocal {
        Vec4 pos{};
        float expansion[ExpansionTerms]{};
    };

    struct alignas(16) GpuEosMeta {
        int32_t offset = 0;
        int32_t resolution = 0;
        int32_t _pad0 = 0;
        int32_t _pad1 = 0;
    };

    struct PushConstants {
        int32_t nParticles = 0;
        int32_t stepTicks = 1;
        int32_t globalTime = 0;
        float minDt = 0.001f;
        float gravityG = 6.67e-5f;
        float boxCenter = 318.0f;
        float boxSize = 318.0f;
        float gravitySoftening = 0.12f;
        int32_t workItems = 0;
        int32_t parentStride = 0;
        int32_t stride = 0;
        int32_t sortIteration = 0;
        int32_t cellTableSize = 0;
        int32_t uncheckedParentStride = 0;
        int32_t gravityLevelParity = 0;
        int32_t dispatchOffset = 0;
        int32_t multipoleChunkElements = 1;
        int32_t localChunkElements = 1;
        int32_t localGravChunkElements = 1;
    };

    enum PipelineIndex : size_t {
        HashPipeline = 0,
        SortHistPipeline = 1,
        SortScanPipeline = 2,
        SortSumPipeline = 3,
        SortScatterPipeline = 4,
        FindCellsPipeline = 5,
        ActivatePipeline = 6,
        GravityUpPipeline = 7,
        GravityDownPipeline = 8,
        DensityPipeline = 9,
        AccelerationPipeline = 10,
        AccelerationStepPipeline = 11,
        IntegratePipeline = 12,
        ShufflePipeline = 13,
        InverseCellsPipeline = 14,
        ShuffleTreePipeline = 15,
        PipelineCount = 16,
    };

    void createInstance();
    void pickPhysicalDevice();
    void createDevice();
    void createCommandResources();
    void createBuffers();
    void createPipeline();
    void updateDescriptorSet();
    void uploadInitialState();
    void rebuildOctreeBuffers();
    void applyOctreeData(const OctreeData& octree);
    void applyPendingOctreeBuild(bool profileStep);
    void startAsyncOctreeBuild();
    void destroyBuffer(Buffer& buffer);
    void destroyChunkedBuffer(ChunkedBuffer& buffer);
    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    ChunkedBuffer createChunkedBuffer(VkDeviceSize elementCount,
                                      VkDeviceSize elementSize,
                                      VkBufferUsageFlags usage,
                                      VkMemoryPropertyFlags properties,
                                      const char* name);
    void copyBufferBlocking(const Buffer& src, const Buffer& dst, VkDeviceSize size, const char* what);
    void uploadBuffer(const Buffer& dst, const void* data, VkDeviceSize size, const char* what);
    void downloadBuffer(const Buffer& src, void* data, VkDeviceSize size, const char* what);
    void downloadChunkedBuffer(const ChunkedBuffer& src, void* data, VkDeviceSize size, const char* what);
    uint32_t findMemory(uint32_t typeBits,
                        VkMemoryPropertyFlags flags,
                        VkMemoryPropertyFlags preferredFlags = 0,
                        VkDeviceSize allocationSize = 0) const;
    VkShaderModule shaderModule(const std::string& path) const;
    void check(VkResult result, const char* what) const;

    Simulator& simulator_;
    uint32_t particleCount_ = 0;
    int32_t globalTimeTicks_ = 0;
    uint64_t diagnosticStepIndex_ = 0;
    std::string deviceName_;
    bool preferDeviceLocalHostVisible_ = false;
    bool segmentSimulationSubmits_ = false;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    bool ownsInstance_ = true;
    bool ownsDevice_ = true;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkCommandBuffer octreeReadbackCommandBuffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkFence octreeReadbackFence_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    std::array<VkPipeline, PipelineCount> pipelines_{};
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

    Buffer positions_;
    Buffer velocities_;
    Buffer densities_;
    Buffer internalEnergy_;
    Buffer masses_;
    Buffer smoothingLengths_;
    Buffer pressures_;
    Buffer materialIds_;
    Buffer temperatures_;
    Buffer alive_;
    Buffer octreePositionsReadback_;
    Buffer octreeAliveReadback_;
    Buffer accelerations_;
    Buffer gravAccelerations_;
    Buffer dInternalEnergy_;
    Buffer dhDt_;
    Buffer tree_;
    ChunkedBuffer multipoles_;
    ChunkedBuffer locals_;
    Buffer parentIndexes_;
    Buffer treeLevel_;
    ChunkedBuffer localGravA_;
    ChunkedBuffer localGravB_;
    Buffer gravAbs_;
    Buffer cellArrayA_;
    Buffer cellArrayB_;
    Buffer largeParticleCells_;
    Buffer bucketHist_;
    Buffer bucketOffset_;
    Buffer particleOffset_;
    Buffer cellStart_;
    Buffer cellEnd_;
    Buffer accelerations1_;
    Buffer dInternalEnergy1_;
    Buffer gradientTerms_;
    Buffer rhoGrads_;
    Buffer particleIds_;
    Buffer speedOfSound_;
    Buffer balsara_;
    Buffer alpha_;
    Buffer daDt_;
    Buffer alphaLoc_;
    Buffer pAlphaLoc_;
    Buffer localMaxH_;
    Buffer nextActiveTime_;
    Buffer eosTables_;
    Buffer eosMeta_;
    Buffer stepTicks_;
    Buffer active_;
    Buffer gravityStepTicks_;
    Buffer scratchPositions_;
    Buffer scratchVelocities_;
    Buffer scratchDensities_;
    Buffer scratchInternalEnergy_;
    Buffer scratchMasses_;
    Buffer scratchSmoothingLengths_;
    Buffer scratchPressures_;
    Buffer scratchMaterialIds_;
    Buffer scratchTemperatures_;
    Buffer scratchAlive_;
    Buffer scratchAccelerations_;
    Buffer scratchGravAccelerations_;
    Buffer scratchDInternalEnergy_;
    Buffer scratchDhDt_;
    Buffer scratchGravAbs_;
    Buffer scratchAccelerations1_;
    Buffer scratchDInternalEnergy1_;
    Buffer scratchGradientTerms_;
    Buffer scratchRhoGrads_;
    Buffer scratchSpeedOfSound_;
    Buffer scratchBalsara_;
    Buffer scratchAlpha_;
    Buffer scratchDaDt_;
    Buffer scratchAlphaLoc_;
    Buffer scratchPAlphaLoc_;
    Buffer scratchLocalMaxH_;
    Buffer scratchNextActiveTime_;
    Buffer scratchParticleIds_;

    std::vector<uint32_t> aliveHost_;
    size_t treeCapacity_ = 0;
    size_t nodeCapacity_ = 0;
    size_t levelCapacity_ = 0;
    size_t localGravCapacity_ = 0;
    uint32_t cellTableSize_ = 0;
    int32_t gravityNextActiveTicks_ = 0;
    int32_t gravityPredictionTicks_ = 0;
    bool hasInitialAcceleration_ = false;
    int32_t stepsSinceShuffle_ = 0;
    std::vector<std::vector<int32_t>> treeLevels_;
    std::vector<uint32_t> treeLevelOffsets_;
    std::future<OctreeData> pendingOctree_;
    bool pendingOctreeStale_ = false;
};

} // namespace sph
