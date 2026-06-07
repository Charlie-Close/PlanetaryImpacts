#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace sph {

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    const std::string& deviceName() const { return deviceName_; }
    uint32_t computeQueueFamily() const { return computeQueueFamily_; }
    bool available() const { return instance_ != VK_NULL_HANDLE; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamily_ = 0;
    std::string deviceName_;
};

std::vector<char> readBinaryFile(const std::string& path);

} // namespace sph

