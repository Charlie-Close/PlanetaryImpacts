#include "sph/VulkanSimulation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include "lodepng.h"
#include "sph/Octree.hpp"
#include "sph/Parameters.hpp"
#include "sph/VulkanContext.hpp"

namespace sph {
namespace {

constexpr uint32_t kWorkgroupSize = 256;
constexpr bool kTraceSimulationPasses = false;

struct SnapshotCameraUniform {
    std::array<float, 16> viewProj{};
    std::array<float, 16> lightViewProj{};
    float cameraPos[4]{};
    float params[4]{};
};

template <typename T>
void copyToMapped(const VulkanSimulation::Buffer& buffer, const std::vector<T>& values) {
    if (!values.empty()) std::memcpy(buffer.mapped, values.data(), sizeof(T) * values.size());
}

template <typename T>
void copyFromMapped(std::vector<T>& values, const VulkanSimulation::Buffer& buffer) {
    if (!values.empty()) std::memcpy(values.data(), buffer.mapped, sizeof(T) * values.size());
}

bool hasDeviceExtension(VkPhysicalDevice device, const char* name) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> available(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, available.data());
    return std::any_of(available.begin(), available.end(), [name](const auto& ext) {
        return std::strcmp(ext.extensionName, name) == 0;
    });
}

std::array<float, 16> multiply4x4(const std::array<float, 16>& a, const std::array<float, 16>& b) {
    std::array<float, 16> out{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            out[col * 4 + row] =
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3];
        }
    }
    return out;
}

std::array<float, 16> perspective4x4(float fovRadians, float aspect, float znear, float zfar) {
    const float ys = 1.0f / std::tan(fovRadians * 0.5f);
    const float xs = ys / aspect;
    const float zs = zfar / (znear - zfar);
    return {xs, 0.0f, 0.0f, 0.0f, 0.0f, ys, 0.0f, 0.0f, 0.0f, 0.0f, zs, -1.0f, 0.0f, 0.0f, znear * zs, 0.0f};
}

std::array<float, 16> orthographic4x4(float left, float right, float bottom, float top, float znear, float zfar) {
    return {
        2.0f / (right - left), 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f / (top - bottom), 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f / (znear - zfar), 0.0f,
        -(right + left) / (right - left), -(top + bottom) / (top - bottom), znear / (znear - zfar), 1.0f,
    };
}

std::array<float, 16> lookAt4x4(Vec3 pos, Vec3 forward, Vec3 up) {
    const Vec3 zAxis = normalize(forward) * -1.0f;
    const Vec3 xAxis = normalize(cross(up, zAxis));
    const Vec3 yAxis = cross(zAxis, xAxis);
    const std::array<float, 16> rotation = {
        xAxis.x, yAxis.x, zAxis.x, 0.0f,
        xAxis.y, yAxis.y, zAxis.y, 0.0f,
        xAxis.z, yAxis.z, zAxis.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const std::array<float, 16> translation = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        -pos.x, -pos.y, -pos.z, 1.0f,
    };
    return multiply4x4(rotation, translation);
}

} // namespace

VulkanSimulation::VulkanSimulation(Simulator& simulator) : simulator_(simulator) {
    particleCount_ = static_cast<uint32_t>(simulator_.particleCount());
    if (particleCount_ == 0) throw std::runtime_error("Cannot create a Vulkan simulation with zero particles");
    std::cout << "Creating simulation Vulkan instance..." << std::endl;
    createInstance();
    std::cout << "Picking simulation Vulkan device..." << std::endl;
    pickPhysicalDevice();
    std::cout << "Creating simulation Vulkan device..." << std::endl;
    createDevice();
    std::cout << "Creating simulation command resources..." << std::endl;
    createCommandResources();
    std::cout << "Creating simulation buffers..." << std::endl;
    createBuffers();
    std::cout << "Uploading simulation initial state..." << std::endl;
    uploadInitialState();
    std::cout << "Creating simulation pipelines..." << std::endl;
    createPipeline();
}

VulkanSimulation::VulkanSimulation(Simulator& simulator,
                                   VkPhysicalDevice physicalDevice,
                                   VkDevice device,
                                   VkQueue queue,
                                   uint32_t queueFamily,
                                   std::string deviceName)
    : simulator_(simulator),
      physicalDevice_(physicalDevice),
      device_(device),
      queue_(queue),
      queueFamily_(queueFamily),
      ownsInstance_(false),
      ownsDevice_(false) {
    particleCount_ = static_cast<uint32_t>(simulator_.particleCount());
    if (particleCount_ == 0) throw std::runtime_error("Cannot create a Vulkan simulation with zero particles");
    deviceName_ = std::move(deviceName);
    const bool hasPortabilitySubset = hasDeviceExtension(physicalDevice_, "VK_KHR_portability_subset");
    preferDeviceLocalHostVisible_ = !hasPortabilitySubset;
    segmentSimulationSubmits_ = hasPortabilitySubset;
    std::cout << "Creating simulation command resources..." << std::endl;
    createCommandResources();
    std::cout << "Creating simulation buffers..." << std::endl;
    createBuffers();
    std::cout << "Uploading simulation initial state..." << std::endl;
    uploadInitialState();
    std::cout << "Creating simulation pipelines..." << std::endl;
    createPipeline();
}

VulkanSimulation::~VulkanSimulation() {
    // All simulation submissions are synchronized with fence_ before step() returns.
    // MoltenVK can wedge indefinitely in a redundant process-exit vkDeviceWaitIdle()
    // after long compute runs, so keep the global idle wait opt-in for diagnostics.
    if (device_ && std::getenv("SPH_WAIT_IDLE_ON_DESTROY") != nullptr) vkDeviceWaitIdle(device_);
    if (pendingOctree_.valid()) pendingOctree_.wait();
    destroyBuffer(octreeAliveReadback_);
    destroyBuffer(octreePositionsReadback_);
    destroyBuffer(alive_);
    destroyBuffer(temperatures_);
    destroyBuffer(materialIds_);
    destroyBuffer(pressures_);
    destroyBuffer(smoothingLengths_);
    destroyBuffer(masses_);
    destroyBuffer(internalEnergy_);
    destroyBuffer(densities_);
    destroyBuffer(velocities_);
    destroyBuffer(positions_);
    destroyBuffer(dhDt_);
    destroyBuffer(dInternalEnergy_);
    destroyBuffer(gravAccelerations_);
    destroyBuffer(accelerations_);
    destroyBuffer(gravAbs_);
    destroyBuffer(localGravB_);
    destroyBuffer(localGravA_);
    destroyBuffer(treeLevel_);
    destroyBuffer(parentIndexes_);
    destroyBuffer(locals_);
    destroyBuffer(multipoles_);
    destroyBuffer(tree_);
    destroyBuffer(cellEnd_);
    destroyBuffer(cellStart_);
    destroyBuffer(particleOffset_);
    destroyBuffer(bucketOffset_);
    destroyBuffer(bucketHist_);
    destroyBuffer(largeParticleCells_);
    destroyBuffer(cellArrayB_);
    destroyBuffer(cellArrayA_);
    destroyBuffer(eosMeta_);
    destroyBuffer(eosTables_);
    destroyBuffer(active_);
    destroyBuffer(stepTicks_);
    destroyBuffer(gravityStepTicks_);
    destroyBuffer(scratchParticleIds_);
    destroyBuffer(scratchNextActiveTime_);
    destroyBuffer(scratchLocalMaxH_);
    destroyBuffer(scratchPAlphaLoc_);
    destroyBuffer(scratchAlphaLoc_);
    destroyBuffer(scratchDaDt_);
    destroyBuffer(scratchAlpha_);
    destroyBuffer(scratchBalsara_);
    destroyBuffer(scratchSpeedOfSound_);
    destroyBuffer(scratchRhoGrads_);
    destroyBuffer(scratchGradientTerms_);
    destroyBuffer(scratchDInternalEnergy1_);
    destroyBuffer(scratchAccelerations1_);
    destroyBuffer(scratchGravAbs_);
    destroyBuffer(scratchDhDt_);
    destroyBuffer(scratchDInternalEnergy_);
    destroyBuffer(scratchGravAccelerations_);
    destroyBuffer(scratchAccelerations_);
    destroyBuffer(scratchAlive_);
    destroyBuffer(scratchTemperatures_);
    destroyBuffer(scratchMaterialIds_);
    destroyBuffer(scratchPressures_);
    destroyBuffer(scratchSmoothingLengths_);
    destroyBuffer(scratchMasses_);
    destroyBuffer(scratchInternalEnergy_);
    destroyBuffer(scratchDensities_);
    destroyBuffer(scratchVelocities_);
    destroyBuffer(scratchPositions_);
    destroyBuffer(nextActiveTime_);
    destroyBuffer(localMaxH_);
    destroyBuffer(pAlphaLoc_);
    destroyBuffer(alphaLoc_);
    destroyBuffer(daDt_);
    destroyBuffer(alpha_);
    destroyBuffer(balsara_);
    destroyBuffer(speedOfSound_);
    destroyBuffer(particleIds_);
    destroyBuffer(rhoGrads_);
    destroyBuffer(gradientTerms_);
    destroyBuffer(dInternalEnergy1_);
    destroyBuffer(accelerations1_);
    if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    for (VkPipeline pipeline : pipelines_) {
        if (pipeline) vkDestroyPipeline(device_, pipeline, nullptr);
    }
    if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    if (descriptorSetLayout_) vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
    if (octreeReadbackFence_) vkDestroyFence(device_, octreeReadbackFence_, nullptr);
    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
    if (ownsDevice_ && device_) vkDestroyDevice(device_, nullptr);
    if (ownsInstance_ && instance_) vkDestroyInstance(instance_, nullptr);
}

void VulkanSimulation::check(VkResult result, const char* what) const {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed");
}

void VulkanSimulation::createInstance() {
    uint32_t extensionCount = 0;
    check(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr), "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    check(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()), "vkEnumerateInstanceExtensionProperties");
    std::vector<const char*> instanceExtensions;
    const bool hasPortabilityEnumeration = std::any_of(availableExtensions.begin(), availableExtensions.end(), [](const auto& ext) {
        return std::string(ext.extensionName) == VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    });
    if (hasPortabilityEnumeration) instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "sph_vulkan_simulation";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "sph_vulkan";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    info.ppEnabledExtensionNames = instanceExtensions.data();
    if (hasPortabilityEnumeration) info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    check(vkCreateInstance(&info, nullptr, &instance_), "vkCreateInstance");
}

void VulkanSimulation::pickPhysicalDevice() {
    uint32_t physicalCount = 0;
    check(vkEnumeratePhysicalDevices(instance_, &physicalCount, nullptr), "vkEnumeratePhysicalDevices");
    if (physicalCount == 0) throw std::runtime_error("No Vulkan physical devices found");
    std::vector<VkPhysicalDevice> devices(physicalCount);
    check(vkEnumeratePhysicalDevices(instance_, &physicalCount, devices.data()), "vkEnumeratePhysicalDevices");
    for (VkPhysicalDevice candidate : devices) {
        uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueCount, queues.data());
        for (uint32_t i = 0; i < queueCount; ++i) {
            if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                physicalDevice_ = candidate;
                queueFamily_ = i;
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(physicalDevice_, &props);
                deviceName_ = props.deviceName;
                return;
            }
        }
    }
    throw std::runtime_error("No Vulkan graphics+compute queue was found");
}

void VulkanSimulation::createDevice() {
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    std::vector<const char*> extensions;
    const bool hasPortabilitySubset = hasDeviceExtension(physicalDevice_, "VK_KHR_portability_subset");
    if (hasPortabilitySubset) extensions.push_back("VK_KHR_portability_subset");
    preferDeviceLocalHostVisible_ = !hasPortabilitySubset;
    segmentSimulationSubmits_ = hasPortabilitySubset;

    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queueInfo;
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physicalDevice_, &info, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
}

void VulkanSimulation::createCommandResources() {
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily_;
    check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = commandPool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device_, &alloc, &commandBuffer_), "vkAllocateCommandBuffers");
    check(vkAllocateCommandBuffers(device_, &alloc, &octreeReadbackCommandBuffer_), "vkAllocateCommandBuffers octree readback");

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device_, &fenceInfo, nullptr, &fence_), "vkCreateFence");
    check(vkCreateFence(device_, &fenceInfo, nullptr, &octreeReadbackFence_), "vkCreateFence octree readback");
}

uint32_t VulkanSimulation::findMemory(uint32_t typeBits, VkMemoryPropertyFlags flags, VkMemoryPropertyFlags preferredFlags) const {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &props);
    if (preferredFlags != 0) {
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & preferredFlags) == preferredFlags) return i;
        }
    }
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("No compatible Vulkan memory type found");
}

VulkanSimulation::Buffer VulkanSimulation::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    Buffer out;
    out.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_, &info, nullptr, &out.buffer), "vkCreateBuffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, out.buffer, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    const bool preferDeviceLocal = preferDeviceLocalHostVisible_ &&
        (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0 &&
        (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    const VkMemoryPropertyFlags preferred = preferDeviceLocal ? (properties | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) : 0;
    alloc.memoryTypeIndex = findMemory(req.memoryTypeBits, properties, preferred);
    check(vkAllocateMemory(device_, &alloc, nullptr, &out.memory), "vkAllocateMemory");
    check(vkBindBufferMemory(device_, out.buffer, out.memory, 0), "vkBindBufferMemory");
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        check(vkMapMemory(device_, out.memory, 0, size, 0, &out.mapped), "vkMapMemory");
    }
    return out;
}

void VulkanSimulation::destroyBuffer(Buffer& buffer) {
    if (buffer.mapped) vkUnmapMemory(device_, buffer.memory);
    if (buffer.buffer) vkDestroyBuffer(device_, buffer.buffer, nullptr);
    if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
    buffer = {};
}

void VulkanSimulation::createBuffers() {
    const VkMemoryPropertyFlags hostCoherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto n = static_cast<VkDeviceSize>(particleCount_);
    positions_ = createBuffer(n * sizeof(Vec3), storage, hostCoherent);
    velocities_ = createBuffer(n * sizeof(Vec3), storage, hostCoherent);
    densities_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    internalEnergy_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    masses_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    smoothingLengths_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    pressures_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    materialIds_ = createBuffer(n * sizeof(int32_t), storage, hostCoherent);
    temperatures_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    alive_ = createBuffer(n * sizeof(uint32_t), storage, hostCoherent);
    octreePositionsReadback_ = createBuffer(n * sizeof(Vec3), VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostCoherent);
    octreeAliveReadback_ = createBuffer(n * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostCoherent);
    accelerations_ = createBuffer(n * sizeof(Vec3), storage, hostCoherent);
    gravAccelerations_ = createBuffer(n * sizeof(Vec3), storage, hostCoherent);
    dInternalEnergy_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    dhDt_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    gravAbs_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    cellArrayA_ = createBuffer(n * sizeof(uint32_t) * 2, storage, hostCoherent);
    cellArrayB_ = createBuffer(n * sizeof(uint32_t) * 2, storage, hostCoherent);
    largeParticleCells_ = createBuffer(n * sizeof(uint32_t), storage, hostCoherent);
    const uint32_t nBlocks = (particleCount_ / params::sortingBlockSize) + 1;
    bucketHist_ = createBuffer(static_cast<VkDeviceSize>(nBlocks) * params::sortingBucketNumber * sizeof(uint32_t), storage, hostCoherent);
    bucketOffset_ = createBuffer(params::sortingBucketNumber * sizeof(uint32_t), storage, hostCoherent);
    particleOffset_ = createBuffer(n * sizeof(uint32_t), storage, hostCoherent);
    cellTableSize_ = 1u << (3 * params::cellPower);
    const VkDeviceSize cellCount = cellTableSize_;
    cellStart_ = createBuffer(cellCount * sizeof(uint32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    cellEnd_ = createBuffer(cellCount * sizeof(uint32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    accelerations1_ = createBuffer(n * sizeof(Vec3), storage, hostCoherent);
    dInternalEnergy1_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    gradientTerms_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    rhoGrads_ = createBuffer(n * sizeof(Vec3), storage, hostCoherent);
    particleIds_ = createBuffer(n * sizeof(int32_t), storage, hostCoherent);
    speedOfSound_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    balsara_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    alpha_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    daDt_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    alphaLoc_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    pAlphaLoc_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    localMaxH_ = createBuffer(n * sizeof(float), storage, hostCoherent);
    nextActiveTime_ = createBuffer(n * sizeof(int32_t), storage, hostCoherent);
    stepTicks_ = createBuffer(sizeof(int32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostCoherent);
    active_ = createBuffer(n * sizeof(uint32_t), storage, hostCoherent);
    gravityStepTicks_ = createBuffer(sizeof(int32_t), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostCoherent);
    const VkBufferUsageFlags shuffleScratchUsage = storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    scratchPositions_ = createBuffer(n * sizeof(Vec3), shuffleScratchUsage, hostCoherent);
    scratchVelocities_ = createBuffer(n * sizeof(Vec3), shuffleScratchUsage, hostCoherent);
    scratchDensities_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchInternalEnergy_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchMasses_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchSmoothingLengths_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchPressures_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchMaterialIds_ = createBuffer(n * sizeof(int32_t), shuffleScratchUsage, hostCoherent);
    scratchTemperatures_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchAlive_ = createBuffer(n * sizeof(uint32_t), shuffleScratchUsage, hostCoherent);
    scratchAccelerations_ = createBuffer(n * sizeof(Vec3), shuffleScratchUsage, hostCoherent);
    scratchGravAccelerations_ = createBuffer(n * sizeof(Vec3), shuffleScratchUsage, hostCoherent);
    scratchDInternalEnergy_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchDhDt_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchGravAbs_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchAccelerations1_ = createBuffer(n * sizeof(Vec3), shuffleScratchUsage, hostCoherent);
    scratchDInternalEnergy1_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchGradientTerms_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchRhoGrads_ = createBuffer(n * sizeof(Vec3), shuffleScratchUsage, hostCoherent);
    scratchSpeedOfSound_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchBalsara_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchAlpha_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchDaDt_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchAlphaLoc_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchPAlphaLoc_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchLocalMaxH_ = createBuffer(n * sizeof(float), shuffleScratchUsage, hostCoherent);
    scratchNextActiveTime_ = createBuffer(n * sizeof(int32_t), shuffleScratchUsage, hostCoherent);
    scratchParticleIds_ = createBuffer(n * sizeof(int32_t), shuffleScratchUsage, hostCoherent);

    const EquationOfState& eos = simulator_.equationOfState();
    const std::array<const AneosTable*, 6> eosTables = {&eos.iron, &eos.forsterite, &eos.fe85si15, &eos.hhe, &eos.ice, &eos.rock};
    size_t eosValueCount = 0;
    for (const AneosTable* table : eosTables) eosValueCount += table->data.size();
    eosTables_ = createBuffer(std::max<VkDeviceSize>(1, static_cast<VkDeviceSize>(eosValueCount)) * sizeof(Vec4), storage, hostCoherent);
    eosMeta_ = createBuffer(eosTables.size() * sizeof(GpuEosMeta), storage, hostCoherent);
}

void VulkanSimulation::uploadInitialState() {
    const DataSet& data = simulator_.data();
    copyToMapped(positions_, data.positions);
    copyToMapped(velocities_, data.velocities);
    copyToMapped(densities_, data.densities);
    copyToMapped(internalEnergy_, data.internalEnergy);
    copyToMapped(masses_, data.masses);
    copyToMapped(smoothingLengths_, data.smoothingLengths);
    copyToMapped(pressures_, data.pressures);
    copyToMapped(materialIds_, data.materialIds);
    copyToMapped(particleIds_, data.particleIds);
    copyToMapped(temperatures_, data.temperatures);
    aliveHost_.assign(particleCount_, 1u);
    std::vector<Vec3> zeros3(particleCount_);
    std::vector<float> zeros(particleCount_, 0.0f);
    for (uint32_t i = 0; i < particleCount_; ++i) {
        Vec3 translated = data.positions[i] - makeVec3(params::boxCenter, params::boxCenter, params::boxCenter);
        aliveHost_[i] = (std::abs(translated.x) <= params::boxSize && std::abs(translated.y) <= params::boxSize && std::abs(translated.z) <= params::boxSize) ? 1u : 0u;
    }
    std::memcpy(alive_.mapped, aliveHost_.data(), aliveHost_.size() * sizeof(uint32_t));
    copyToMapped(accelerations_, zeros3);
    copyToMapped(gravAccelerations_, zeros3);
    copyToMapped(dInternalEnergy_, zeros);
    copyToMapped(dhDt_, zeros);
    copyToMapped(gravAbs_, zeros);
    copyToMapped(accelerations1_, zeros3);
    copyToMapped(dInternalEnergy1_, zeros);
    std::vector<float> ones(particleCount_, 1.0f);
    std::vector<float> alpha(particleCount_, params::viscosityAlpha);
    std::vector<int32_t> nextActive(particleCount_, 0);
    std::vector<uint32_t> active(particleCount_, 1u);
    int32_t initialStepTicks = 1;
    copyToMapped(gradientTerms_, ones);
    copyToMapped(rhoGrads_, zeros3);
    copyToMapped(speedOfSound_, ones);
    copyToMapped(balsara_, ones);
    copyToMapped(alpha_, alpha);
    copyToMapped(daDt_, zeros);
    copyToMapped(alphaLoc_, alpha);
    copyToMapped(pAlphaLoc_, alpha);
    copyToMapped(localMaxH_, data.smoothingLengths);
    copyToMapped(nextActiveTime_, nextActive);
    copyToMapped(active_, active);
    copyToMapped(stepTicks_, std::vector<int32_t>{initialStepTicks});
    copyToMapped(gravityStepTicks_, std::vector<int32_t>{initialStepTicks});

    const EquationOfState& eos = simulator_.equationOfState();
    const std::array<const AneosTable*, 6> eosTables = {&eos.iron, &eos.forsterite, &eos.fe85si15, &eos.hhe, &eos.ice, &eos.rock};
    std::vector<GpuEosMeta> eosMeta(eosTables.size());
    Vec4* eosMapped = static_cast<Vec4*>(eosTables_.mapped);
    size_t eosOffset = 0;
    for (size_t i = 0; i < eosTables.size(); ++i) {
        eosMeta[i].offset = static_cast<int32_t>(eosOffset);
        eosMeta[i].resolution = eosTables[i]->resolution;
        if (!eosTables[i]->data.empty()) {
            std::memcpy(eosMapped + eosOffset, eosTables[i]->data.data(), eosTables[i]->data.size() * sizeof(Vec4));
        }
        eosOffset += eosTables[i]->data.size();
    }
    copyToMapped(eosMeta_, eosMeta);
    rebuildOctreeBuffers();
}

void VulkanSimulation::rebuildOctreeBuffers() {
    std::vector<Vec3> positions(particleCount_);
    copyFromMapped(positions, positions_);
    std::memcpy(aliveHost_.data(), alive_.mapped, aliveHost_.size() * sizeof(uint32_t));
    const OctreeData octree = buildOctree(positions, aliveHost_, 8, std::max<size_t>(treeCapacity_, positions.size() * 2));
    applyOctreeData(octree);
}

void VulkanSimulation::applyOctreeData(const OctreeData& octree) {
    treeLevels_ = octree.levels;

    const VkMemoryPropertyFlags hostCoherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (octree.tree.size() > treeCapacity_) {
        destroyBuffer(tree_);
        treeCapacity_ = static_cast<size_t>(static_cast<double>(octree.tree.size()) * 1.1) + 64;
        tree_ = createBuffer(treeCapacity_ * sizeof(int32_t), storage, hostCoherent);
    }
    if (static_cast<size_t>(octree.nodeValues) > nodeCapacity_) {
        destroyBuffer(multipoles_);
        destroyBuffer(locals_);
        destroyBuffer(parentIndexes_);
        nodeCapacity_ = static_cast<size_t>(static_cast<double>(octree.nodeValues) * 1.1) + 64;
        multipoles_ = createBuffer(nodeCapacity_ * sizeof(GpuMultipole), storage, hostCoherent);
        locals_ = createBuffer(nodeCapacity_ * sizeof(GpuLocal), storage, hostCoherent);
        parentIndexes_ = createBuffer(nodeCapacity_ * sizeof(uint32_t), storage, hostCoherent);
    }
    size_t totalLevelEntries = 0;
    size_t maxLevel = 0;
    treeLevelOffsets_.clear();
    treeLevelOffsets_.reserve(treeLevels_.size());
    for (const auto& level : treeLevels_) {
        treeLevelOffsets_.push_back(static_cast<uint32_t>(totalLevelEntries));
        totalLevelEntries += level.size();
        maxLevel = std::max(maxLevel, level.size());
    }
    if (totalLevelEntries > levelCapacity_) {
        destroyBuffer(treeLevel_);
        levelCapacity_ = static_cast<size_t>(static_cast<double>(totalLevelEntries) * 1.1) + 64;
        treeLevel_ = createBuffer(levelCapacity_ * sizeof(int32_t), storage, hostCoherent);
    }
    int32_t* levelMapped = static_cast<int32_t*>(treeLevel_.mapped);
    for (size_t level = 0; level < treeLevels_.size(); ++level) {
        if (!treeLevels_[level].empty()) {
            std::memcpy(levelMapped + treeLevelOffsets_[level], treeLevels_[level].data(), treeLevels_[level].size() * sizeof(int32_t));
        }
    }
    const size_t localGravSize = std::max<size_t>(1, maxLevel * MaxUncheckedPointers);
    if (localGravSize > localGravCapacity_) {
        destroyBuffer(localGravA_);
        destroyBuffer(localGravB_);
        localGravCapacity_ = static_cast<size_t>(static_cast<double>(localGravSize) * 1.1) + 64;
        localGravA_ = createBuffer(localGravCapacity_ * sizeof(int32_t), storage, hostCoherent);
        localGravB_ = createBuffer(localGravCapacity_ * sizeof(int32_t), storage, hostCoherent);
    }
    if (!octree.tree.empty()) std::memcpy(tree_.mapped, octree.tree.data(), octree.tree.size() * sizeof(int32_t));
    if (descriptorSet_ != VK_NULL_HANDLE) updateDescriptorSet();
}

void VulkanSimulation::applyPendingOctreeBuild(bool profileStep) {
    if (!pendingOctree_.valid()) return;
    if (pendingOctree_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    const auto applyStart = std::chrono::steady_clock::now();
    OctreeData octree = pendingOctree_.get();
    if (!pendingOctreeStale_) {
        applyOctreeData(octree);
    }
    pendingOctreeStale_ = false;
    if (profileStep) {
        const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - applyStart;
        std::cout << "[profile] cpu octree apply: " << elapsed.count() << " ms\n";
    }
}

void VulkanSimulation::startAsyncOctreeBuild() {
    const bool profileStep = std::getenv("SPH_PROFILE_STEPS") != nullptr;
    const auto totalStart = std::chrono::steady_clock::now();
    if (pendingOctree_.valid()) {
        if (pendingOctree_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            if (profileStep) std::cout << "[profile] cpu octree pending: still building\n";
            return;
        }
        const auto applyStart = std::chrono::steady_clock::now();
        OctreeData octree = pendingOctree_.get();
        if (!pendingOctreeStale_) {
            applyOctreeData(octree);
        }
        pendingOctreeStale_ = false;
        if (profileStep) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - applyStart;
            std::cout << "[profile] cpu octree finish/apply: " << elapsed.count() << " ms\n";
        }
    }
    const auto readStart = std::chrono::steady_clock::now();
    check(vkResetFences(device_, 1, &octreeReadbackFence_), "vkResetFences octree readback");
    check(vkResetCommandBuffer(octreeReadbackCommandBuffer_, 0), "vkResetCommandBuffer octree readback");
    VkCommandBufferBeginInfo readBegin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    readBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(octreeReadbackCommandBuffer_, &readBegin), "vkBeginCommandBuffer octree readback");
    VkBufferCopy positionsCopy{};
    positionsCopy.size = positions_.size;
    vkCmdCopyBuffer(octreeReadbackCommandBuffer_, positions_.buffer, octreePositionsReadback_.buffer, 1, &positionsCopy);
    VkBufferCopy aliveCopy{};
    aliveCopy.size = alive_.size;
    vkCmdCopyBuffer(octreeReadbackCommandBuffer_, alive_.buffer, octreeAliveReadback_.buffer, 1, &aliveCopy);
    check(vkEndCommandBuffer(octreeReadbackCommandBuffer_), "vkEndCommandBuffer octree readback");
    VkSubmitInfo readSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    readSubmit.commandBufferCount = 1;
    readSubmit.pCommandBuffers = &octreeReadbackCommandBuffer_;
    check(vkQueueSubmit(queue_, 1, &readSubmit, octreeReadbackFence_), "vkQueueSubmit octree readback");
    if (profileStep) {
        const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - readStart;
        std::cout << "[profile] cpu octree readback submit: " << elapsed.count() << " ms\n";
    }
    const size_t estimatedTreeSize = std::max<size_t>(treeCapacity_, static_cast<size_t>(particleCount_) * 2);
    pendingOctree_ = std::async(std::launch::async, [this, estimatedTreeSize, profileStep]() {
        const auto asyncStart = std::chrono::steady_clock::now();
        const VkResult waitResult = vkWaitForFences(device_, 1, &octreeReadbackFence_, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS) throw std::runtime_error("vkWaitForFences octree async readback failed");
        std::vector<Vec3> positions(particleCount_);
        std::vector<uint32_t> alive(aliveHost_.size());
        std::memcpy(positions.data(), octreePositionsReadback_.mapped, positions.size() * sizeof(Vec3));
        std::memcpy(alive.data(), octreeAliveReadback_.mapped, alive.size() * sizeof(uint32_t));
        if (profileStep) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - asyncStart;
            std::cout << "[profile] cpu octree async readback wait/copy: " << elapsed.count() << " ms\n";
        }
        return buildOctree(positions, alive, 8, estimatedTreeSize);
    });
    if (profileStep) {
        const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - totalStart;
        std::cout << "[profile] cpu octree schedule total: " << elapsed.count() << " ms\n";
    }
}

VkShaderModule VulkanSimulation::shaderModule(const std::string& path) const {
    std::vector<char> bytes = readBinaryFile(path);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes.size();
    info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device_, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

void VulkanSimulation::updateDescriptorSet() {
    if (descriptorSet_ == VK_NULL_HANDLE) return;
    std::array<Buffer*, 76> buffers = {&positions_, &velocities_, &densities_, &internalEnergy_, &masses_,
                                       &smoothingLengths_, &pressures_, &materialIds_, &temperatures_, &alive_,
                                       &accelerations_, &gravAccelerations_, &dInternalEnergy_, &dhDt_,
                                       &tree_, &multipoles_, &locals_, &parentIndexes_, &treeLevel_,
                                       &localGravA_, &localGravB_, &gravAbs_,
                                       &cellArrayA_, &cellArrayB_, &largeParticleCells_, &bucketHist_,
                                       &bucketOffset_, &particleOffset_, &cellStart_, &cellEnd_,
                                       &accelerations1_, &dInternalEnergy1_, &gradientTerms_, &speedOfSound_,
                                       &balsara_, &alpha_, &daDt_, &alphaLoc_, &pAlphaLoc_, &localMaxH_,
                                       &nextActiveTime_, &eosTables_, &eosMeta_, &stepTicks_, &active_,
                                       &gravityStepTicks_, &rhoGrads_,
                                       &scratchPositions_, &scratchVelocities_, &scratchDensities_, &scratchInternalEnergy_,
                                       &scratchMasses_, &scratchSmoothingLengths_, &scratchPressures_, &scratchMaterialIds_,
                                       &scratchTemperatures_, &scratchAlive_, &scratchAccelerations_, &scratchGravAccelerations_,
                                       &scratchDInternalEnergy_, &scratchDhDt_, &scratchGravAbs_, &scratchAccelerations1_,
                                       &scratchDInternalEnergy1_, &scratchGradientTerms_, &scratchRhoGrads_, &scratchSpeedOfSound_,
                                       &scratchBalsara_, &scratchAlpha_, &scratchDaDt_, &scratchAlphaLoc_,
                                       &scratchPAlphaLoc_, &scratchLocalMaxH_, &scratchNextActiveTime_,
                                       &particleIds_, &scratchParticleIds_};
    std::array<VkDescriptorBufferInfo, 76> infos{};
    std::array<VkWriteDescriptorSet, 76> writes{};
    for (uint32_t i = 0; i < buffers.size(); ++i) {
        infos[i].buffer = buffers[i]->buffer;
        infos[i].offset = 0;
        infos[i].range = buffers[i]->size;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptorSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void VulkanSimulation::createPipeline() {
    std::cout << "Creating simulation descriptor layout..." << std::endl;
    std::array<VkDescriptorSetLayoutBinding, 76> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_), "vkCreateDescriptorSetLayout");

    std::cout << "Creating simulation pipeline layout..." << std::endl;
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_), "vkCreatePipelineLayout");

    std::cout << "Creating simulation compute pipelines..." << std::endl;
    const std::array<const char*, PipelineCount> shaderPaths = {
        "build/shaders/sph_hash.comp.spv",
        "build/shaders/sph_sort_hist.comp.spv",
        "build/shaders/sph_sort_scan.comp.spv",
        "build/shaders/sph_sort_sum.comp.spv",
        "build/shaders/sph_sort_scatter.comp.spv",
        "build/shaders/sph_find_cells.comp.spv",
        "build/shaders/activate.comp.spv",
        "build/shaders/sph_gravity_up.comp.spv",
        "build/shaders/sph_gravity_down.comp.spv",
        "build/shaders/sph_density.comp.spv",
        "build/shaders/sph_acceleration.comp.spv",
        "build/shaders/sph_acceleration_step.comp.spv",
        "build/shaders/sph_integrate.comp.spv",
        "build/shaders/sph_shuffle.comp.spv",
        "build/shaders/sph_inverse_cells.comp.spv",
        "build/shaders/sph_shuffle_tree.comp.spv",
    };
    for (size_t i = 0; i < shaderPaths.size(); ++i) {
        std::cout << "  pipeline " << shaderPaths[i] << std::endl;
        VkShaderModule shader = shaderModule(shaderPaths[i]);
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = stage;
        pipelineInfo.layout = pipelineLayout_;
        check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelines_[i]), "vkCreateComputePipelines");
        vkDestroyShaderModule(device_, shader, nullptr);
    }

    std::cout << "Creating simulation descriptor set..." << std::endl;
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(bindings.size());
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = descriptorPool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &descriptorSetLayout_;
    check(vkAllocateDescriptorSets(device_, &alloc, &descriptorSet_), "vkAllocateDescriptorSets");

    std::cout << "Writing simulation descriptors..." << std::endl;
    updateDescriptorSet();
    std::cout << "Simulation pipeline setup complete." << std::endl;
}

void VulkanSimulation::step(int maxTicks) {
    maxTicks = std::max(maxTicks, 0);
    const uint64_t diagnosticStepIndex = ++diagnosticStepIndex_;
    const bool profileStep = std::getenv("SPH_PROFILE_STEPS") != nullptr;
    const bool debugStats = std::getenv("SPH_DEBUG_STATS") != nullptr;
    const uint64_t traceStart = std::getenv("SPH_TRACE_STEP_START") != nullptr
        ? static_cast<uint64_t>(std::strtoull(std::getenv("SPH_TRACE_STEP_START"), nullptr, 10))
        : 0u;
    const uint64_t traceEnd = std::getenv("SPH_TRACE_STEP_END") != nullptr
        ? static_cast<uint64_t>(std::strtoull(std::getenv("SPH_TRACE_STEP_END"), nullptr, 10))
        : std::numeric_limits<uint64_t>::max();
    const bool traceSegments = std::getenv("SPH_TRACE_SEGMENTS") != nullptr &&
        diagnosticStepIndex >= traceStart && diagnosticStepIndex <= traceEnd;
    const bool traceGravityBuffers = std::getenv("SPH_TRACE_GRAVITY_BUFFERS") != nullptr &&
        diagnosticStepIndex >= traceStart && diagnosticStepIndex <= traceEnd;
    const bool traceGravityStats = std::getenv("SPH_TRACE_GRAVITY_STATS") != nullptr &&
        diagnosticStepIndex >= traceStart && diagnosticStepIndex <= traceEnd;
    const char* slowSegmentEnv = std::getenv("SPH_SLOW_SEGMENT_MS");
    const double slowSegmentMs = slowSegmentEnv != nullptr ? std::atof(slowSegmentEnv) : 0.0;
    const char* gpuWaitTimeoutEnv = std::getenv("SPH_GPU_WAIT_TIMEOUT_MS");
    const uint64_t gpuWaitTimeoutMs = gpuWaitTimeoutEnv != nullptr
        ? static_cast<uint64_t>(std::strtoull(gpuWaitTimeoutEnv, nullptr, 10))
        : 0u;
    uint32_t gravityDownChunkSize = 8192 * 4;
    if (const char* chunkEnv = std::getenv("SPH_GRAVITY_CHUNK_SIZE")) {
        gravityDownChunkSize = std::max<uint32_t>(64u, static_cast<uint32_t>(std::atoi(chunkEnv)));
    }
    applyPendingOctreeBuild(profileStep);
    const int32_t stepStartTicks = globalTimeTicks_;
    const bool runGravity = globalTimeTicks_ >= gravityNextActiveTicks_;
    if (runGravity) {
        if (kTraceSimulationPasses) std::cout << "Using current octree buffers..." << std::endl;
    }
    if (kTraceSimulationPasses) std::cout << "Recording GPU simulation step..." << std::endl;
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    check(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(commandBuffer_, &begin), "vkBeginCommandBuffer");
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
    bool recordedCommands = false;
    PushConstants pc{};
    pc.nParticles = static_cast<int32_t>(particleCount_);
    pc.stepTicks = maxTicks > 0 ? maxTicks : 1;
    pc.globalTime = globalTimeTicks_;
    pc.minDt = params::minDt;
    pc.gravityG = params::gravityG;
    pc.boxCenter = static_cast<float>(params::boxCenter);
    pc.boxSize = static_cast<float>(params::boxSize);
    pc.gravitySoftening = params::gravitySmoothingLength;
    pc.cellTableSize = static_cast<int32_t>(cellTableSize_);
    auto barrier = [&] {
        VkMemoryBarrier memoryBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    };
    auto dispatch = [&](VkPipeline pipeline, uint32_t workItems, uint32_t localSize = kWorkgroupSize, uint32_t logicalWorkItems = 0) {
        pc.workItems = static_cast<int32_t>(logicalWorkItems == 0 ? workItems : logicalWorkItems);
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(commandBuffer_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);
        vkCmdDispatch(commandBuffer_, (workItems + localSize - 1) / localSize, 1, 1);
        recordedCommands = true;
        barrier();
    };
    auto waitForSimulationFence = [&](const std::string& label) {
        if (gpuWaitTimeoutMs == 0u) {
            check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), label.c_str());
            return;
        }
        const uint64_t timeoutNs = gpuWaitTimeoutMs * 1000000ull;
        const VkResult result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeoutNs);
        if (result == VK_SUCCESS) return;
        if (result == VK_TIMEOUT) {
            throw std::runtime_error(label + " timed out after " + std::to_string(gpuWaitTimeoutMs) +
                                     " ms at simulation step " + std::to_string(diagnosticStepIndex) +
                                     " time_ticks=" + std::to_string(globalTimeTicks_));
        }
        check(result, label.c_str());
    };
    const bool forceSegmentSubmits = segmentSimulationSubmits_ ||
        profileStep || traceSegments || traceGravityBuffers || traceGravityStats || slowSegmentMs > 0.0;
    auto submitProfileSegment = [&](const char* label) {
        if (!forceSegmentSubmits) return;
        check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer profile segment");
        const auto submitStart = std::chrono::steady_clock::now();
        if (recordedCommands) {
            if (traceSegments) {
                std::cout << "[trace] step=" << diagnosticStepIndex
                          << " ticks=" << globalTimeTicks_
                          << " submit " << label << std::endl;
            }
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &commandBuffer_;
            const std::string submitLabel = std::string("vkQueueSubmit ") + label;
            const std::string waitLabel = std::string("vkWaitForFences ") + label;
            check(vkQueueSubmit(queue_, 1, &submit, fence_), submitLabel.c_str());
            waitForSimulationFence(waitLabel);
            if (traceSegments) {
                std::cout << "[trace] step=" << diagnosticStepIndex
                          << " ticks=" << globalTimeTicks_
                          << " done " << label << std::endl;
            }
        }
        const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - submitStart;
        if (profileStep || (slowSegmentMs > 0.0 && elapsed.count() >= slowSegmentMs)) {
            std::cout << "[profile] gpu " << label << ": " << elapsed.count() << " ms\n";
        }
        check(vkResetFences(device_, 1, &fence_), "vkResetFences profile segment");
        check(vkResetCommandBuffer(commandBuffer_, 0), "vkResetCommandBuffer profile segment");
        check(vkBeginCommandBuffer(commandBuffer_, &begin), "vkBeginCommandBuffer profile segment");
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
        recordedCommands = false;
    };

    const uint32_t nBlocks = (particleCount_ / params::sortingBlockSize) + 1;
    if (kTraceSimulationPasses) std::cout << "  hash" << std::endl;
    dispatch(pipelines_[HashPipeline], particleCount_);
    for (int iteration = 0; iteration < 4; ++iteration) {
        pc.sortIteration = iteration;
        if (kTraceSimulationPasses) std::cout << "  sort iteration " << iteration << std::endl;
        dispatch(pipelines_[SortHistPipeline], nBlocks, 1);
        dispatch(pipelines_[SortScanPipeline], params::sortingBucketNumber, 1, nBlocks);
        dispatch(pipelines_[SortSumPipeline], 1, 1);
        dispatch(pipelines_[SortScatterPipeline], particleCount_);
        VkBufferCopy copy{};
        copy.size = cellArrayA_.size;
        vkCmdCopyBuffer(commandBuffer_, cellArrayB_.buffer, cellArrayA_.buffer, 1, &copy);
        barrier();
    }
    vkCmdFillBuffer(commandBuffer_, cellStart_.buffer, 0, cellStart_.size, 0xFFFFFFFFu);
    vkCmdFillBuffer(commandBuffer_, cellEnd_.buffer, 0, cellEnd_.size, 0u);
    barrier();
    if (kTraceSimulationPasses) std::cout << "  find cells" << std::endl;
    dispatch(pipelines_[FindCellsPipeline], particleCount_);
    if (kTraceSimulationPasses) std::cout << "  activate" << std::endl;
    dispatch(pipelines_[ActivatePipeline], particleCount_);
    submitProfileSegment("hash/sort/find/activate");

    const bool isolateEarlyPasses = false;
    if (isolateEarlyPasses) {
        check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer_;
        std::cout << "Submitting isolated hash/sort/find pass..." << std::endl;
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        std::cout << "Isolated hash/sort/find pass completed." << std::endl;
        globalTimeTicks_ += std::max(maxTicks, 1);
        simulator_.setGlobalTimeTicks(globalTimeTicks_);
        return;
    }

    const int32_t maxStepTicks = static_cast<int32_t>(params::maxDt / params::minDt);
    int32_t initialStepTicks = maxStepTicks;
    if (!runGravity && gravityNextActiveTicks_ > globalTimeTicks_) {
        initialStepTicks = std::clamp(gravityNextActiveTicks_ - globalTimeTicks_, 1, maxStepTicks);
    }
    if (maxTicks > 0) initialStepTicks = std::min(initialStepTicks, maxTicks);
    vkCmdFillBuffer(commandBuffer_, stepTicks_.buffer, 0, stepTicks_.size, static_cast<uint32_t>(initialStepTicks));
    vkCmdFillBuffer(commandBuffer_, gravityStepTicks_.buffer, 0, gravityStepTicks_.size, static_cast<uint32_t>(maxStepTicks));
    recordedCommands = true;
    barrier();

    if (runGravity) {
        pc.stepTicks = gravityPredictionTicks_;
        if (kTraceSimulationPasses) std::cout << "  gravity up" << std::endl;
        for (size_t level = treeLevels_.size(); level-- > 0;) {
            if (treeLevels_[level].empty()) continue;
            pc.parentStride = static_cast<int32_t>(treeLevelOffsets_[level]);
            dispatch(pipelines_[GravityUpPipeline], static_cast<uint32_t>(treeLevels_[level].size()));
        }
        submitProfileSegment("gravity up");
        if (traceGravityStats) {
            const auto* multipoles = static_cast<const GpuMultipole*>(multipoles_.mapped);
            const auto* tree = static_cast<const int32_t*>(tree_.mapped);
            const auto* gravAbs = static_cast<const float*>(gravAbs_.mapped);
            uint32_t gravZero = 0;
            uint32_t gravNonFinite = 0;
            float gravMin = std::numeric_limits<float>::max();
            float gravMax = 0.0f;
            double gravSum = 0.0;
            for (uint32_t p = 0; p < particleCount_; ++p) {
                const float g = gravAbs[p];
                if (!std::isfinite(g)) {
                    ++gravNonFinite;
                    continue;
                }
                if (g <= 0.0f) ++gravZero;
                gravMin = std::min(gravMin, g);
                gravMax = std::max(gravMax, g);
                gravSum += g;
            }
            std::cout << "[gravstats] step=" << diagnosticStepIndex
                      << " particles=" << particleCount_
                      << " gravAbs_min=" << gravMin
                      << " gravAbs_max=" << gravMax
                      << " gravAbs_mean=" << (particleCount_ > 0 ? gravSum / static_cast<double>(particleCount_) : 0.0)
                      << " gravAbs_zero=" << gravZero
                      << " gravAbs_nonfinite=" << gravNonFinite
                      << "\n";
            for (size_t level = 0; level < treeLevels_.size(); ++level) {
                if (treeLevels_[level].empty()) continue;
                uint32_t branchCount = 0;
                uint32_t leafCount = 0;
                uint32_t badData = 0;
                uint32_t zeroMinGrav = 0;
                uint32_t nonFiniteMinGrav = 0;
                uint32_t nonFinitePower = 0;
                float minMinGrav = std::numeric_limits<float>::max();
                float maxMinGrav = 0.0f;
                float minSize = std::numeric_limits<float>::max();
                float maxSize = 0.0f;
                float maxPower3 = 0.0f;
                double meanSize = 0.0;
                double meanPower3 = 0.0;
                for (int32_t treePointer : treeLevels_[level]) {
                    if (treePointer < 0 || static_cast<size_t>(treePointer + 1) >= treeCapacity_) {
                        ++badData;
                        continue;
                    }
                    const int32_t nParticles = tree[treePointer];
                    const int32_t dataPointer = tree[treePointer + 1];
                    if (dataPointer < 0 || static_cast<size_t>(dataPointer) >= nodeCapacity_) {
                        ++badData;
                        continue;
                    }
                    if (nParticles == 0) ++branchCount;
                    else ++leafCount;
                    const GpuMultipole& mp = multipoles[dataPointer];
                    if (!std::isfinite(mp.minGrav)) ++nonFiniteMinGrav;
                    else {
                        if (mp.minGrav <= 0.0f) ++zeroMinGrav;
                        minMinGrav = std::min(minMinGrav, mp.minGrav);
                        maxMinGrav = std::max(maxMinGrav, mp.minGrav);
                    }
                    if (std::isfinite(mp.size)) {
                        minSize = std::min(minSize, mp.size);
                        maxSize = std::max(maxSize, mp.size);
                        meanSize += mp.size;
                    }
                    if (std::isfinite(mp.power[3])) {
                        maxPower3 = std::max(maxPower3, mp.power[3]);
                        meanPower3 += mp.power[3];
                    } else {
                        ++nonFinitePower;
                    }
                }
                const double nodes = static_cast<double>(treeLevels_[level].size());
                std::cout << "[gravstats] step=" << diagnosticStepIndex
                          << " level=" << level
                          << " nodes=" << treeLevels_[level].size()
                          << " branches=" << branchCount
                          << " leaves=" << leafCount
                          << " bad=" << badData
                          << " minGrav=" << minMinGrav << ".." << maxMinGrav
                          << " zeroMinGrav=" << zeroMinGrav
                          << " nonfiniteMinGrav=" << nonFiniteMinGrav
                          << " size=" << minSize << ".." << maxSize
                          << " meanSize=" << (nodes > 0.0 ? meanSize / nodes : 0.0)
                          << " power3Max=" << maxPower3
                          << " power3Mean=" << (nodes > 0.0 ? meanPower3 / nodes : 0.0)
                          << " nonfinitePower=" << nonFinitePower
                          << "\n";
            }
        }
    } else {
        if (kTraceSimulationPasses) std::cout << "  gravity skipped" << std::endl;
    }
    const bool isolateGravityUpOnly = false;
    if (isolateGravityUpOnly) {
        check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer_;
        std::cout << "Submitting isolated hash/sort/gravity-up pass..." << std::endl;
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        std::cout << "Isolated hash/sort/gravity-up pass completed." << std::endl;
        globalTimeTicks_ += std::max(maxTicks, 1);
        simulator_.setGlobalTimeTicks(globalTimeTicks_);
        return;
    }
    if (runGravity) {
        pc.stepTicks = gravityPredictionTicks_;
        if (kTraceSimulationPasses) std::cout << "  gravity down" << std::endl;
        int32_t uncheckedParentStride = 0;
        for (size_t level = 0; level < treeLevels_.size(); ++level) {
            if (treeLevels_[level].empty()) continue;
            const uint32_t levelSize = static_cast<uint32_t>(treeLevels_[level].size());
            const int32_t uncheckedStride = std::max<int32_t>(
                1, static_cast<int32_t>(localGravCapacity_ / std::max<size_t>(1, treeLevels_[level].size())));
            pc.parentStride = static_cast<int32_t>(treeLevelOffsets_[level]);
            pc.stride = uncheckedStride;
            pc.uncheckedParentStride = uncheckedParentStride;
            pc.gravityLevelParity = static_cast<int32_t>(level & 1u);
            for (uint32_t offset = 0; offset < levelSize; offset += gravityDownChunkSize) {
                pc.dispatchOffset = static_cast<int32_t>(offset);
                const uint32_t chunkSize = std::min(gravityDownChunkSize, levelSize - offset);
                dispatch(pipelines_[GravityDownPipeline], chunkSize, kWorkgroupSize, levelSize);
                const std::string label = "gravity down level " + std::to_string(level) +
                    " offset " + std::to_string(offset);
                submitProfileSegment(label.c_str());
            }
            if (traceGravityBuffers) {
                const Buffer& outBuffer = (level & 1u) == 0u ? localGravB_ : localGravA_;
                const auto* out = static_cast<const int32_t*>(outBuffer.mapped);
                const auto* tree = static_cast<const int32_t*>(tree_.mapped);
                const auto* parentIndexes = static_cast<const uint32_t*>(parentIndexes_.mapped);
                uint32_t noSentinel = 0;
                uint32_t invalidEntries = 0;
                int32_t maxUsed = 0;
                const uint32_t sampleCount = std::min<uint32_t>(levelSize, 8u);
                std::cout << "[gravbuf] step=" << diagnosticStepIndex
                          << " level=" << level
                          << " nodes=" << levelSize
                          << " stride=" << uncheckedStride
                          << " parent_stride=" << uncheckedParentStride
                          << " samples=";
                for (uint32_t n = 0; n < levelSize; ++n) {
                    const int32_t base = static_cast<int32_t>(n) * uncheckedStride;
                    int32_t firstSentinel = uncheckedStride;
                    for (int32_t s = 0; s < uncheckedStride; ++s) {
                        const int32_t value = out[base + s];
                        if (value == -1) {
                            firstSentinel = s;
                            break;
                        }
                        if (value < 0 || static_cast<size_t>(value + 1) >= treeCapacity_ || tree[value + 1] < 0) {
                            ++invalidEntries;
                        }
                    }
                    if (firstSentinel == uncheckedStride) ++noSentinel;
                    maxUsed = std::max(maxUsed, firstSentinel);
                    if (n < sampleCount) {
                        if (n != 0) std::cout << ",";
                        std::cout << firstSentinel;
                    }
                }
                std::cout << " max_used=" << maxUsed
                          << " no_sentinel=" << noSentinel
                          << " invalid_entries=" << invalidEntries;
                if (level + 1 < treeLevels_.size() && !treeLevels_[level + 1].empty()) {
                    uint32_t minParent = std::numeric_limits<uint32_t>::max();
                    uint32_t maxParent = 0;
                    uint32_t badParent = 0;
                    for (int32_t childTreePointer : treeLevels_[level + 1]) {
                        if (childTreePointer < 0 || static_cast<size_t>(childTreePointer + 1) >= treeCapacity_) {
                            ++badParent;
                            continue;
                        }
                        const int32_t childData = tree[childTreePointer + 1];
                        if (childData < 0 || static_cast<size_t>(childData) >= nodeCapacity_) {
                            ++badParent;
                            continue;
                        }
                        const uint32_t parent = parentIndexes[childData];
                        minParent = std::min(minParent, parent);
                        maxParent = std::max(maxParent, parent);
                        if (parent >= levelSize) ++badParent;
                    }
                    std::cout << " next_parent_range=" << minParent << ".." << maxParent
                              << " bad_next_parent=" << badParent;
                }
                std::cout << "\n";
            }
            uncheckedParentStride = uncheckedStride;
        }
    }
    pc.stride = 0;
    pc.uncheckedParentStride = 0;
    pc.gravityLevelParity = 0;
    pc.dispatchOffset = 0;
    pc.stepTicks = maxTicks > 0 ? maxTicks : 1;
    const bool isolateGravityPasses = false;
    if (isolateGravityPasses) {
        check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer_;
        std::cout << "Submitting isolated hash/sort/gravity pass..." << std::endl;
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        std::cout << "Isolated hash/sort/gravity pass completed." << std::endl;
        globalTimeTicks_ += std::max(maxTicks, 1);
        simulator_.setGlobalTimeTicks(globalTimeTicks_);
        return;
    }
    if (!hasInitialAcceleration_) {
        for (int i = 0; i < params::densityGradientSettlingIterations; ++i) {
            if (kTraceSimulationPasses) std::cout << "  density settling " << i << std::endl;
            dispatch(pipelines_[DensityPipeline], particleCount_);
            submitProfileSegment("density settling");
        }
    }
    if (kTraceSimulationPasses) std::cout << "  density" << std::endl;
    dispatch(pipelines_[DensityPipeline], particleCount_);
    submitProfileSegment("density");
    if (kTraceSimulationPasses) std::cout << "  acceleration" << std::endl;
    dispatch(pipelines_[AccelerationPipeline], particleCount_);
    submitProfileSegment("acceleration");
    if (hasInitialAcceleration_) {
        if (kTraceSimulationPasses) std::cout << "  acceleration step" << std::endl;
        dispatch(pipelines_[AccelerationStepPipeline], particleCount_);
        if (kTraceSimulationPasses) std::cout << "  integrate" << std::endl;
        dispatch(pipelines_[IntegratePipeline], particleCount_);
        submitProfileSegment("acceleration step/integrate");
    } else if (kTraceSimulationPasses) {
        std::cout << "  initial acceleration seeded; integration deferred" << std::endl;
    }
    const bool runShuffle = stepsSinceShuffle_ > params::shuffleFrames;
    if (runShuffle) {
        if (kTraceSimulationPasses) std::cout << "  shuffle" << std::endl;
        dispatch(pipelines_[ShufflePipeline], particleCount_);
        auto copyWhole = [&](const Buffer& src, const Buffer& dst) {
            VkBufferCopy copy{};
            copy.size = std::min(src.size, dst.size);
            vkCmdCopyBuffer(commandBuffer_, src.buffer, dst.buffer, 1, &copy);
            recordedCommands = true;
            barrier();
        };
        copyWhole(scratchPositions_, positions_);
        copyWhole(scratchVelocities_, velocities_);
        copyWhole(scratchDensities_, densities_);
        copyWhole(scratchInternalEnergy_, internalEnergy_);
        copyWhole(scratchMasses_, masses_);
        copyWhole(scratchSmoothingLengths_, smoothingLengths_);
        copyWhole(scratchPressures_, pressures_);
        copyWhole(scratchMaterialIds_, materialIds_);
        copyWhole(scratchTemperatures_, temperatures_);
        copyWhole(scratchAlive_, alive_);
        copyWhole(scratchAccelerations_, accelerations_);
        copyWhole(scratchGravAccelerations_, gravAccelerations_);
        copyWhole(scratchDInternalEnergy_, dInternalEnergy_);
        copyWhole(scratchDhDt_, dhDt_);
        copyWhole(scratchGravAbs_, gravAbs_);
        copyWhole(scratchAccelerations1_, accelerations1_);
        copyWhole(scratchDInternalEnergy1_, dInternalEnergy1_);
        copyWhole(scratchGradientTerms_, gradientTerms_);
        copyWhole(scratchRhoGrads_, rhoGrads_);
        copyWhole(scratchSpeedOfSound_, speedOfSound_);
        copyWhole(scratchBalsara_, balsara_);
        copyWhole(scratchAlpha_, alpha_);
        copyWhole(scratchDaDt_, daDt_);
        copyWhole(scratchAlphaLoc_, alphaLoc_);
        copyWhole(scratchPAlphaLoc_, pAlphaLoc_);
        copyWhole(scratchLocalMaxH_, localMaxH_);
        copyWhole(scratchNextActiveTime_, nextActiveTime_);
        copyWhole(scratchParticleIds_, particleIds_);
        if (pendingOctree_.valid()) pendingOctreeStale_ = true;
        dispatch(pipelines_[InverseCellsPipeline], particleCount_);
        for (size_t level = 0; level < treeLevels_.size(); ++level) {
            if (treeLevels_[level].empty()) continue;
            pc.parentStride = static_cast<int32_t>(treeLevelOffsets_[level]);
            dispatch(pipelines_[ShuffleTreePipeline], static_cast<uint32_t>(treeLevels_[level].size()));
        }
        pc.parentStride = 0;
        submitProfileSegment("shuffle/copy-back/tree");
    }
    check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer");

    if (kTraceSimulationPasses) std::cout << "Submitting GPU simulation step..." << std::endl;
    if (recordedCommands) {
        const auto submitStart = std::chrono::steady_clock::now();
        if (traceSegments) {
            std::cout << "[trace] step=" << diagnosticStepIndex
                      << " ticks=" << globalTimeTicks_
                      << " submit final segment" << std::endl;
        }
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer_;
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
        waitForSimulationFence("vkWaitForFences final segment");
        if (traceSegments) {
            std::cout << "[trace] step=" << diagnosticStepIndex
                      << " ticks=" << globalTimeTicks_
                      << " done final segment" << std::endl;
        }
        if (profileStep) {
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - submitStart;
            std::cout << "[profile] gpu final segment: " << elapsed.count() << " ms\n";
        }
    }
    if (kTraceSimulationPasses) std::cout << "GPU simulation step completed." << std::endl;
    if (debugStats) {
        auto printScalarStats = [&](const char* name, const Buffer& buffer) {
            const float* values = static_cast<const float*>(buffer.mapped);
            std::vector<float> finite;
            finite.reserve(particleCount_);
            for (uint32_t i = 0; i < particleCount_; ++i) {
                if (std::isfinite(values[i])) finite.push_back(values[i]);
            }
            if (finite.empty()) {
                std::cout << "[stats] " << name << ": no finite values\n";
                return;
            }
            std::sort(finite.begin(), finite.end());
            auto pct = [&](double p) {
                const size_t idx = std::min(finite.size() - 1, static_cast<size_t>(p * static_cast<double>(finite.size() - 1)));
                return finite[idx];
            };
            std::cout << "[stats] " << name
                      << " min=" << finite.front()
                      << " p50=" << pct(0.50)
                      << " p99=" << pct(0.99)
                      << " max=" << finite.back() << "\n";
        };
        auto printVecStats = [&](const char* name, const Buffer& buffer) {
            const Vec3* values = static_cast<const Vec3*>(buffer.mapped);
            std::vector<float> finite;
            finite.reserve(particleCount_);
            for (uint32_t i = 0; i < particleCount_; ++i) {
                const float mag = length(values[i]);
                if (std::isfinite(mag)) finite.push_back(mag);
            }
            if (finite.empty()) {
                std::cout << "[stats] " << name << ": no finite values\n";
                return;
            }
            std::sort(finite.begin(), finite.end());
            auto pct = [&](double p) {
                const size_t idx = std::min(finite.size() - 1, static_cast<size_t>(p * static_cast<double>(finite.size() - 1)));
                return finite[idx];
            };
            std::cout << "[stats] " << name
                      << " min=" << finite.front()
                      << " p50=" << pct(0.50)
                      << " p99=" << pct(0.99)
                      << " max=" << finite.back() << "\n";
        };
        printScalarStats("density", densities_);
        printScalarStats("pressure", pressures_);
        printScalarStats("sound_speed", speedOfSound_);
        printScalarStats("temperature", temperatures_);
        printScalarStats("h", smoothingLengths_);
        printVecStats("rho_grad", rhoGrads_);
        printVecStats("hydro_acc", accelerations_);
        printVecStats("grav_acc", gravAccelerations_);
        printVecStats("velocity", velocities_);
        const auto* active = static_cast<const uint32_t*>(active_.mapped);
        uint32_t activeCount = 0;
        for (uint32_t i = 0; i < particleCount_; ++i) activeCount += active[i] != 0u ? 1u : 0u;
        const auto* cells = static_cast<const uint32_t*>(cellArrayA_.mapped);
        const auto* positions = static_cast<const Vec3*>(positions_.mapped);
        auto splitBy3 = [](uint32_t a) {
            uint32_t x = a & 0x000003ffu;
            x = (x | (x << 16)) & 0x030000FFu;
            x = (x | (x << 8)) & 0x0300F00Fu;
            x = (x | (x << 4)) & 0x030C30C3u;
            x = (x | (x << 2)) & 0x09249249u;
            return x;
        };
        auto morton3D = [&](uint32_t x, uint32_t y, uint32_t z) {
            return splitBy3(x) | (splitBy3(y) << 1u) | (splitBy3(z) << 2u);
        };
        auto cellKeyForParticle = [&](uint32_t particle) {
            const int mask = (1 << params::cellPower) - 1;
            const Vec3& p = positions[particle];
            const int x = static_cast<int>(std::floor(p.x / params::cellWidth));
            const int y = static_cast<int>(std::floor(p.y / params::cellWidth));
            const int z = static_cast<int>(std::floor(p.z / params::cellWidth));
            return morton3D(static_cast<uint32_t>(x & mask),
                            static_cast<uint32_t>(y & mask),
                            static_cast<uint32_t>(z & mask));
        };
        uint32_t descents = 0;
        uint32_t duplicateKeys = 0;
        uint32_t badParticleIds = 0;
        uint32_t mismatchedKeys = 0;
        uint32_t duplicateParticleIds = 0;
        std::vector<uint8_t> seen(particleCount_, 0u);
        for (uint32_t i = 0; i < particleCount_; ++i) {
            const uint32_t key = cells[2u * i + 0u];
            const uint32_t particle = cells[2u * i + 1u];
            if (particle >= particleCount_) {
                ++badParticleIds;
            } else {
                if (seen[particle] != 0u) ++duplicateParticleIds;
                seen[particle] = 1u;
                if (key != cellKeyForParticle(particle)) ++mismatchedKeys;
            }
            if (i > 0) {
                const uint32_t prev = cells[2u * (i - 1u) + 0u];
                if (prev > key) ++descents;
                if (prev == key) ++duplicateKeys;
            }
        }
        std::cout << "[stats] active=" << activeCount << "/" << particleCount_
                  << " cell_descents=" << descents
                  << " duplicate_adjacent_cell_keys=" << duplicateKeys
                  << " bad_cell_particle_ids=" << badParticleIds
                  << " duplicate_particle_ids=" << duplicateParticleIds
                  << " mismatched_cell_keys=" << mismatchedKeys;
        if (particleCount_ > 0) {
            std::cout << " first_cell=(" << cells[0] << "," << cells[1] << ")"
                      << " last_cell=(" << cells[2u * (particleCount_ - 1u)] << ","
                      << cells[2u * (particleCount_ - 1u) + 1u] << ")";
        }
        std::cout << "\n";
    }
    int32_t completedTicks = 0;
    std::memcpy(&completedTicks, stepTicks_.mapped, sizeof(completedTicks));
    completedTicks = std::max(completedTicks, 1);
    if (maxTicks > 0) completedTicks = std::min(completedTicks, maxTicks);
    if (runGravity) {
        int32_t gravityTicks = 0;
        std::memcpy(&gravityTicks, gravityStepTicks_.mapped, sizeof(gravityTicks));
        gravityTicks = std::clamp(gravityTicks, 1, static_cast<int32_t>(params::maxDt / params::minDt));
        gravityNextActiveTicks_ = stepStartTicks + gravityTicks;
        gravityPredictionTicks_ = gravityTicks;
    }
    startAsyncOctreeBuild();
    if (hasInitialAcceleration_) {
        globalTimeTicks_ += completedTicks;
    } else {
        hasInitialAcceleration_ = true;
    }
    if (runShuffle) stepsSinceShuffle_ = 0;
    else ++stepsSinceShuffle_;
    simulator_.setGlobalTimeTicks(globalTimeTicks_);
}

void VulkanSimulation::writeSnapshot(const std::filesystem::path& path, Vec3 cameraPosition, float pitch, float yaw) {
    constexpr uint32_t snapshotSize = static_cast<uint32_t>(params::snapshotResolution);
    constexpr uint32_t shadowSize = 1024;
    const VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    const VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    auto createImage = [&](uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory) {
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = {width, height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateImage(device_, &imageInfo, nullptr, &image), "vkCreateImage snapshot");
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device_, image, &req);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device_, &alloc, nullptr, &memory), "vkAllocateMemory snapshot image");
        check(vkBindImageMemory(device_, image, memory, 0), "vkBindImageMemory snapshot");
    };

    auto createView = [&](VkImage image, VkFormat format, VkImageAspectFlags aspect) {
        VkImageView view = VK_NULL_HANDLE;
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        check(vkCreateImageView(device_, &viewInfo, nullptr, &view), "vkCreateImageView snapshot");
        return view;
    };

    Buffer quadBuffer{};
    Buffer indexBuffer{};
    Buffer instanceIdBuffer{};
    Buffer visibleCountBuffer{};
    Buffer indirectDrawBuffer{};
    Buffer uniformBuffer{};
    Buffer readbackBuffer{};
    VkImage colorImage = VK_NULL_HANDLE;
    VkDeviceMemory colorMemory = VK_NULL_HANDLE;
    VkImageView colorView = VK_NULL_HANDLE;
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;
    VkImage shadowImage = VK_NULL_HANDLE;
    VkDeviceMemory shadowMemory = VK_NULL_HANDLE;
    VkImageView shadowView = VK_NULL_HANDLE;
    VkSampler shadowSampler = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkFramebuffer shadowFramebuffer = VK_NULL_HANDLE;
    VkDescriptorSetLayout renderSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout packSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet renderSet = VK_NULL_HANDLE;
    VkDescriptorSet packSet = VK_NULL_HANDLE;
    VkPipelineLayout renderPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout packPipelineLayout = VK_NULL_HANDLE;
    VkPipeline renderPipeline = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipeline packPipeline = VK_NULL_HANDLE;

    auto cleanup = [&] {
        if (device_) vkDeviceWaitIdle(device_);
        if (packPipeline) vkDestroyPipeline(device_, packPipeline, nullptr);
        if (shadowPipeline) vkDestroyPipeline(device_, shadowPipeline, nullptr);
        if (renderPipeline) vkDestroyPipeline(device_, renderPipeline, nullptr);
        if (packPipelineLayout) vkDestroyPipelineLayout(device_, packPipelineLayout, nullptr);
        if (shadowPipelineLayout) vkDestroyPipelineLayout(device_, shadowPipelineLayout, nullptr);
        if (renderPipelineLayout) vkDestroyPipelineLayout(device_, renderPipelineLayout, nullptr);
        if (descriptorPool) vkDestroyDescriptorPool(device_, descriptorPool, nullptr);
        if (packSetLayout) vkDestroyDescriptorSetLayout(device_, packSetLayout, nullptr);
        if (renderSetLayout) vkDestroyDescriptorSetLayout(device_, renderSetLayout, nullptr);
        if (shadowFramebuffer) vkDestroyFramebuffer(device_, shadowFramebuffer, nullptr);
        if (framebuffer) vkDestroyFramebuffer(device_, framebuffer, nullptr);
        if (shadowRenderPass) vkDestroyRenderPass(device_, shadowRenderPass, nullptr);
        if (renderPass) vkDestroyRenderPass(device_, renderPass, nullptr);
        if (shadowSampler) vkDestroySampler(device_, shadowSampler, nullptr);
        if (shadowView) vkDestroyImageView(device_, shadowView, nullptr);
        if (shadowImage) vkDestroyImage(device_, shadowImage, nullptr);
        if (shadowMemory) vkFreeMemory(device_, shadowMemory, nullptr);
        if (depthView) vkDestroyImageView(device_, depthView, nullptr);
        if (depthImage) vkDestroyImage(device_, depthImage, nullptr);
        if (depthMemory) vkFreeMemory(device_, depthMemory, nullptr);
        if (colorView) vkDestroyImageView(device_, colorView, nullptr);
        if (colorImage) vkDestroyImage(device_, colorImage, nullptr);
        if (colorMemory) vkFreeMemory(device_, colorMemory, nullptr);
        destroyBuffer(readbackBuffer);
        destroyBuffer(uniformBuffer);
        destroyBuffer(indirectDrawBuffer);
        destroyBuffer(visibleCountBuffer);
        destroyBuffer(instanceIdBuffer);
        destroyBuffer(indexBuffer);
        destroyBuffer(quadBuffer);
    };

    try {
        const VkMemoryPropertyFlags hostCoherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const std::array<float, 8> quad = {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
        quadBuffer = createBuffer(sizeof(quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, hostCoherent);
        std::memcpy(quadBuffer.mapped, quad.data(), sizeof(quad));
        const std::array<uint16_t, 6> indices = {0, 1, 2, 0, 2, 3};
        indexBuffer = createBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, hostCoherent);
        std::memcpy(indexBuffer.mapped, indices.data(), sizeof(indices));
        instanceIdBuffer = createBuffer(static_cast<VkDeviceSize>(particleCount_) * sizeof(uint32_t),
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        visibleCountBuffer = createBuffer(sizeof(uint32_t),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        indirectDrawBuffer = createBuffer(5 * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        uniformBuffer = createBuffer(sizeof(SnapshotCameraUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostCoherent);
        readbackBuffer = createBuffer(static_cast<VkDeviceSize>(snapshotSize) * snapshotSize * 4,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, hostCoherent);

        const float yawRad = yaw * 3.14159265358979323846f / 180.0f;
        const float pitchRad = pitch * 3.14159265358979323846f / 180.0f;
        const Vec3 forward = normalize(makeVec3(std::cos(yawRad) * std::cos(pitchRad), std::sin(pitchRad), std::sin(yawRad) * std::cos(pitchRad)));
        SnapshotCameraUniform uniform{};
        uniform.viewProj = multiply4x4(perspective4x4(45.0f * 3.14159265358979323846f / 180.0f, 1.0f, 10.0f, 1000.0f),
                                       lookAt4x4(cameraPosition, forward, makeVec3(0.0f, 1.0f, 0.0f)));
        const Vec3 lightDir = normalize(params::lightDirection);
        const Vec3 lightPos = makeVec3(params::boxCenter, params::boxCenter, params::boxCenter) - lightDir * 300.0f;
        uniform.lightViewProj = multiply4x4(orthographic4x4(-50.0f, 50.0f, -50.0f, 50.0f, 0.0f, 500.0f),
                                            lookAt4x4(lightPos, lightDir, makeVec3(0.0f, 1.0f, 0.0f)));
        uniform.cameraPos[0] = cameraPosition.x;
        uniform.cameraPos[1] = cameraPosition.y;
        uniform.cameraPos[2] = cameraPosition.z;
        uniform.params[0] = params::particleSize;
        std::memcpy(uniformBuffer.mapped, &uniform, sizeof(uniform));

        createImage(snapshotSize, snapshotSize, colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, colorImage, colorMemory);
        colorView = createView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        createImage(snapshotSize, snapshotSize, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImage, depthMemory);
        depthView = createView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
        createImage(shadowSize, shadowSize, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, shadowImage, shadowMemory);
        shadowView = createView(shadowImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = 1.0f;
        check(vkCreateSampler(device_, &sampler, nullptr, &shadowSampler), "vkCreateSampler snapshot shadow");

        VkAttachmentDescription color{};
        color.format = colorFormat;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        VkAttachmentDescription depth{};
        depth.format = depthFormat;
        depth.samples = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;
        std::array<VkAttachmentDescription, 2> attachments = {color, depth};
        VkRenderPassCreateInfo rpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpInfo.pAttachments = attachments.data();
        rpInfo.subpassCount = 1;
        rpInfo.pSubpasses = &subpass;
        check(vkCreateRenderPass(device_, &rpInfo, nullptr, &renderPass), "vkCreateRenderPass snapshot");

        VkAttachmentDescription shadowDepth{};
        shadowDepth.format = depthFormat;
        shadowDepth.samples = VK_SAMPLE_COUNT_1_BIT;
        shadowDepth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        shadowDepth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        shadowDepth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        shadowDepth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        shadowDepth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        shadowDepth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkAttachmentReference shadowDepthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription shadowSubpass{};
        shadowSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        shadowSubpass.pDepthStencilAttachment = &shadowDepthRef;
        VkRenderPassCreateInfo shadowRpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        shadowRpInfo.attachmentCount = 1;
        shadowRpInfo.pAttachments = &shadowDepth;
        shadowRpInfo.subpassCount = 1;
        shadowRpInfo.pSubpasses = &shadowSubpass;
        check(vkCreateRenderPass(device_, &shadowRpInfo, nullptr, &shadowRenderPass), "vkCreateRenderPass snapshot shadow");

        std::array<VkImageView, 2> framebufferAttachments = {colorView, depthView};
        VkFramebufferCreateInfo fbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbInfo.renderPass = renderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(framebufferAttachments.size());
        fbInfo.pAttachments = framebufferAttachments.data();
        fbInfo.width = snapshotSize;
        fbInfo.height = snapshotSize;
        fbInfo.layers = 1;
        check(vkCreateFramebuffer(device_, &fbInfo, nullptr, &framebuffer), "vkCreateFramebuffer snapshot");
        VkFramebufferCreateInfo shadowFbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        shadowFbInfo.renderPass = shadowRenderPass;
        shadowFbInfo.attachmentCount = 1;
        shadowFbInfo.pAttachments = &shadowView;
        shadowFbInfo.width = shadowSize;
        shadowFbInfo.height = shadowSize;
        shadowFbInfo.layers = 1;
        check(vkCreateFramebuffer(device_, &shadowFbInfo, nullptr, &shadowFramebuffer), "vkCreateFramebuffer snapshot shadow");

        std::array<VkDescriptorSetLayoutBinding, 10> renderBindings{};
        renderBindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        renderBindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        for (uint32_t i = 2; i < renderBindings.size(); ++i) {
            renderBindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo renderLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        renderLayoutInfo.bindingCount = static_cast<uint32_t>(renderBindings.size());
        renderLayoutInfo.pBindings = renderBindings.data();
        check(vkCreateDescriptorSetLayout(device_, &renderLayoutInfo, nullptr, &renderSetLayout), "vkCreateDescriptorSetLayout snapshot render");
        std::array<VkDescriptorSetLayoutBinding, 10> packBindings{};
        for (uint32_t i = 0; i < packBindings.size(); ++i) {
            packBindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo packLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        packLayoutInfo.bindingCount = static_cast<uint32_t>(packBindings.size());
        packLayoutInfo.pBindings = packBindings.data();
        check(vkCreateDescriptorSetLayout(device_, &packLayoutInfo, nullptr, &packSetLayout), "vkCreateDescriptorSetLayout snapshot pack");

        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 18};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
        poolSizes[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool snapshot");
        VkDescriptorSetAllocateInfo renderAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        renderAlloc.descriptorPool = descriptorPool;
        renderAlloc.descriptorSetCount = 1;
        renderAlloc.pSetLayouts = &renderSetLayout;
        check(vkAllocateDescriptorSets(device_, &renderAlloc, &renderSet), "vkAllocateDescriptorSets snapshot render");
        VkDescriptorSetAllocateInfo packAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        packAlloc.descriptorPool = descriptorPool;
        packAlloc.descriptorSetCount = 1;
        packAlloc.pSetLayouts = &packSetLayout;
        check(vkAllocateDescriptorSets(device_, &packAlloc, &packSet), "vkAllocateDescriptorSets snapshot pack");

        VkDescriptorBufferInfo uniformInfo{uniformBuffer.buffer, 0, sizeof(SnapshotCameraUniform)};
        VkDescriptorImageInfo shadowImageInfo{shadowSampler, shadowView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        std::array<VkDescriptorBufferInfo, 8> renderStorage{};
        renderStorage[0] = {positions_.buffer, 0, positions_.size};
        renderStorage[1] = {densities_.buffer, 0, densities_.size};
        renderStorage[2] = {smoothingLengths_.buffer, 0, smoothingLengths_.size};
        renderStorage[3] = {temperatures_.buffer, 0, temperatures_.size};
        renderStorage[4] = {materialIds_.buffer, 0, materialIds_.size};
        renderStorage[5] = {rhoGrads_.buffer, 0, rhoGrads_.size};
        renderStorage[6] = {alpha_.buffer, 0, alpha_.size};
        renderStorage[7] = {instanceIdBuffer.buffer, 0, instanceIdBuffer.size};
        std::array<VkWriteDescriptorSet, 10> renderWrites{};
        renderWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        renderWrites[0].dstSet = renderSet;
        renderWrites[0].dstBinding = 0;
        renderWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        renderWrites[0].descriptorCount = 1;
        renderWrites[0].pBufferInfo = &uniformInfo;
        renderWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        renderWrites[1].dstSet = renderSet;
        renderWrites[1].dstBinding = 1;
        renderWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        renderWrites[1].descriptorCount = 1;
        renderWrites[1].pImageInfo = &shadowImageInfo;
        for (uint32_t i = 0; i < renderStorage.size(); ++i) {
            renderWrites[2 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            renderWrites[2 + i].dstSet = renderSet;
            renderWrites[2 + i].dstBinding = 2 + i;
            renderWrites[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            renderWrites[2 + i].descriptorCount = 1;
            renderWrites[2 + i].pBufferInfo = &renderStorage[i];
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(renderWrites.size()), renderWrites.data(), 0, nullptr);

        std::array<VkDescriptorBufferInfo, 10> packInfos{};
        packInfos[0] = {positions_.buffer, 0, positions_.size};
        packInfos[1] = {densities_.buffer, 0, densities_.size};
        packInfos[2] = {smoothingLengths_.buffer, 0, smoothingLengths_.size};
        packInfos[3] = {temperatures_.buffer, 0, temperatures_.size};
        packInfos[4] = {materialIds_.buffer, 0, materialIds_.size};
        packInfos[5] = {rhoGrads_.buffer, 0, rhoGrads_.size};
        packInfos[6] = {alpha_.buffer, 0, alpha_.size};
        packInfos[7] = {instanceIdBuffer.buffer, 0, instanceIdBuffer.size};
        packInfos[8] = {visibleCountBuffer.buffer, 0, visibleCountBuffer.size};
        packInfos[9] = {indirectDrawBuffer.buffer, 0, indirectDrawBuffer.size};
        std::array<VkWriteDescriptorSet, 10> packWrites{};
        for (uint32_t i = 0; i < packWrites.size(); ++i) {
            packWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            packWrites[i].dstSet = packSet;
            packWrites[i].dstBinding = i;
            packWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            packWrites[i].descriptorCount = 1;
            packWrites[i].pBufferInfo = &packInfos[i];
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(packWrites.size()), packWrites.data(), 0, nullptr);

        auto createRenderPipeline = [&](bool shadow) {
            VkShaderModule vert = shaderModule(shadow ? "build/shaders/particle_shadow.vert.spv" : "build/shaders/particle.vert.spv");
            VkShaderModule frag = shaderModule(shadow ? "build/shaders/particle_shadow.frag.spv" : "build/shaders/particle.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vert; stages[0].pName = "main";
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = frag; stages[1].pName = "main";
            std::array<VkVertexInputBindingDescription, 1> bindings{};
            bindings[0] = {0, sizeof(float) * 2, VK_VERTEX_INPUT_RATE_VERTEX};
            std::array<VkVertexInputAttributeDescription, 1> attrs{};
            attrs[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
            VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
            vertexInput.pVertexBindingDescriptions = bindings.data();
            vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
            vertexInput.pVertexAttributeDescriptions = attrs.data();
            VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkViewport viewport{0, 0, static_cast<float>(shadow ? shadowSize : snapshotSize), static_cast<float>(shadow ? shadowSize : snapshotSize), 0, 1};
            VkRect2D scissor{{0, 0}, {shadow ? shadowSize : snapshotSize, shadow ? shadowSize : snapshotSize}};
            VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewportState.viewportCount = 1; viewportState.pViewports = &viewport; viewportState.scissorCount = 1; viewportState.pScissors = &scissor;
            VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL; raster.cullMode = VK_CULL_MODE_NONE; raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo depthState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depthState.depthTestEnable = VK_TRUE;
            depthState.depthWriteEnable = VK_TRUE;
            depthState.depthCompareOp = VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo blendState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blendState.attachmentCount = shadow ? 0u : 1u;
            blendState.pAttachments = shadow ? nullptr : &blend;
            VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout.setLayoutCount = 1;
            layout.pSetLayouts = &renderSetLayout;
            VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
            check(vkCreatePipelineLayout(device_, &layout, nullptr, &pipelineLayout), shadow ? "vkCreatePipelineLayout snapshot shadow" : "vkCreatePipelineLayout snapshot render");
            VkGraphicsPipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipeInfo.stageCount = 2; pipeInfo.pStages = stages; pipeInfo.pVertexInputState = &vertexInput; pipeInfo.pInputAssemblyState = &assembly; pipeInfo.pViewportState = &viewportState;
            pipeInfo.pRasterizationState = &raster; pipeInfo.pMultisampleState = &msaa; pipeInfo.pDepthStencilState = &depthState; pipeInfo.pColorBlendState = &blendState;
            pipeInfo.layout = pipelineLayout; pipeInfo.renderPass = shadow ? shadowRenderPass : renderPass;
            VkPipeline pipeline = VK_NULL_HANDLE;
            check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline), shadow ? "vkCreateGraphicsPipelines snapshot shadow" : "vkCreateGraphicsPipelines snapshot render");
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
            if (shadow) shadowPipelineLayout = pipelineLayout;
            else renderPipelineLayout = pipelineLayout;
            return pipeline;
        };
        renderPipeline = createRenderPipeline(false);
        shadowPipeline = createRenderPipeline(true);

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = 16;
        VkPipelineLayoutCreateInfo packPipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        packPipelineLayoutInfo.setLayoutCount = 1;
        packPipelineLayoutInfo.pSetLayouts = &packSetLayout;
        packPipelineLayoutInfo.pushConstantRangeCount = 1;
        packPipelineLayoutInfo.pPushConstantRanges = &push;
        check(vkCreatePipelineLayout(device_, &packPipelineLayoutInfo, nullptr, &packPipelineLayout), "vkCreatePipelineLayout snapshot pack");
        VkShaderModule packShader = shaderModule("build/shaders/pack_particles.comp.spv");
        VkPipelineShaderStageCreateInfo packStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        packStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        packStage.module = packShader;
        packStage.pName = "main";
        VkComputePipelineCreateInfo packPipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        packPipelineInfo.stage = packStage;
        packPipelineInfo.layout = packPipelineLayout;
        check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &packPipelineInfo, nullptr, &packPipeline), "vkCreateComputePipelines snapshot pack");
        vkDestroyShaderModule(device_, packShader, nullptr);

        vkResetFences(device_, 1, &fence_);
        vkResetCommandBuffer(commandBuffer_, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(commandBuffer_, &begin), "vkBeginCommandBuffer snapshot");
        struct PackPush {
            int32_t count;
            float cameraX;
            float cameraY;
            float cameraZ;
        } packPush{static_cast<int32_t>(particleCount_), cameraPosition.x, cameraPosition.y, cameraPosition.z};
        const uint32_t drawArgs[5] = {6u, 0u, 0u, 0u, 0u};
        vkCmdFillBuffer(commandBuffer_, visibleCountBuffer.buffer, 0, visibleCountBuffer.size, 0u);
        vkCmdUpdateBuffer(commandBuffer_, indirectDrawBuffer.buffer, 0, sizeof(drawArgs), drawArgs);
        VkMemoryBarrier transferBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        transferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        transferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &transferBarrier, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, packPipeline);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, packPipelineLayout, 0, 1, &packSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer_, packPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(packPush), &packPush);
        vkCmdDispatch(commandBuffer_, (particleCount_ + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
        VkMemoryBarrier computeBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        computeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
        VkBufferCopy countCopy{};
        countCopy.dstOffset = sizeof(uint32_t);
        countCopy.size = sizeof(uint32_t);
        vkCmdCopyBuffer(commandBuffer_, visibleCountBuffer.buffer, indirectDrawBuffer.buffer, 1, &countCopy);
        VkMemoryBarrier drawBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        drawBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        drawBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &drawBarrier, 0, nullptr, 0, nullptr);

        VkClearValue shadowClear{};
        shadowClear.depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo shadowBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        shadowBegin.renderPass = shadowRenderPass;
        shadowBegin.framebuffer = shadowFramebuffer;
        shadowBegin.renderArea.extent = {shadowSize, shadowSize};
        shadowBegin.clearValueCount = 1;
        shadowBegin.pClearValues = &shadowClear;
        vkCmdBeginRenderPass(commandBuffer_, &shadowBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
        VkDeviceSize vertexOffset = 0;
        vkCmdBindVertexBuffers(commandBuffer_, 0, 1, &quadBuffer.buffer, &vertexOffset);
        vkCmdBindIndexBuffer(commandBuffer_, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout, 0, 1, &renderSet, 0, nullptr);
        vkCmdDrawIndexedIndirect(commandBuffer_, indirectDrawBuffer.buffer, 0, 1, 0);
        vkCmdEndRenderPass(commandBuffer_);
        VkMemoryBarrier shadowBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        shadowBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        shadowBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer_, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &shadowBarrier, 0, nullptr, 0, nullptr);

        std::array<VkClearValue, 2> clear{};
        clear[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clear[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo renderBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        renderBegin.renderPass = renderPass;
        renderBegin.framebuffer = framebuffer;
        renderBegin.renderArea.extent = {snapshotSize, snapshotSize};
        renderBegin.clearValueCount = static_cast<uint32_t>(clear.size());
        renderBegin.pClearValues = clear.data();
        vkCmdBeginRenderPass(commandBuffer_, &renderBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, renderPipeline);
        vkCmdBindVertexBuffers(commandBuffer_, 0, 1, &quadBuffer.buffer, &vertexOffset);
        vkCmdBindIndexBuffer(commandBuffer_, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, renderPipelineLayout, 0, 1, &renderSet, 0, nullptr);
        vkCmdDrawIndexedIndirect(commandBuffer_, indirectDrawBuffer.buffer, 0, 1, 0);
        vkCmdEndRenderPass(commandBuffer_);

        VkBufferImageCopy imageCopy{};
        imageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopy.imageSubresource.layerCount = 1;
        imageCopy.imageExtent = {snapshotSize, snapshotSize, 1};
        vkCmdCopyImageToBuffer(commandBuffer_, colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer.buffer, 1, &imageCopy);
        check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer snapshot");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer_;
        check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit snapshot");
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences snapshot");

        std::filesystem::create_directories(path.parent_path().empty() ? "." : path.parent_path());
        const auto* rgba = static_cast<const unsigned char*>(readbackBuffer.mapped);
        if (path.extension() == ".rgba") {
            std::ofstream out(path, std::ios::binary | std::ios::app);
            if (!out) throw std::runtime_error("Could not open raw snapshot stream: " + path.string());
            out.write(reinterpret_cast<const char*>(rgba), static_cast<std::streamsize>(static_cast<size_t>(snapshotSize) * snapshotSize * 4u));
            if (!out) throw std::runtime_error("Could not write raw snapshot frame: " + path.string());
        } else {
            std::vector<unsigned char> png;
            unsigned error = lodepng::encode(png, rgba, snapshotSize, snapshotSize);
            if (error) throw std::runtime_error("PNG encode failed: " + std::string(lodepng_error_text(error)));
            lodepng::save_file(png, path.string());
        }
        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

void VulkanSimulation::syncToSimulator() {
    DataSet& data = simulator_.mutableData();
    copyFromMapped(data.positions, positions_);
    copyFromMapped(data.velocities, velocities_);
    copyFromMapped(data.densities, densities_);
    copyFromMapped(data.internalEnergy, internalEnergy_);
    copyFromMapped(data.smoothingLengths, smoothingLengths_);
    copyFromMapped(data.pressures, pressures_);
    copyFromMapped(data.temperatures, temperatures_);
    copyFromMapped(data.particleIds, particleIds_);
}

} // namespace sph
