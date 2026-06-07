#include "sph/VulkanContext.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace sph {
namespace {

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed");
}

} // namespace

VulkanContext::VulkanContext() {
    uint32_t extensionCount = 0;
    check(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr), "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    check(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()), "vkEnumerateInstanceExtensionProperties");
    std::vector<const char*> instanceExtensions;
    const bool hasPortabilityEnumeration = std::any_of(availableExtensions.begin(), availableExtensions.end(), [](const auto& ext) {
        return std::string(ext.extensionName) == VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    });
    if (hasPortabilityEnumeration) {
        instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "sph_vulkan";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "sph_vulkan";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &app;
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
    if (hasPortabilityEnumeration) {
        instanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    check(vkCreateInstance(&instanceInfo, nullptr, &instance_), "vkCreateInstance");

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
            if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice_ = candidate;
                computeQueueFamily_ = i;
                break;
            }
        }
        if (physicalDevice_) break;
    }
    if (!physicalDevice_) throw std::runtime_error("No Vulkan compute queue was found");

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &props);
    deviceName_ = props.deviceName;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamily_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    uint32_t deviceExtensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &deviceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> deviceExtensionsAvailable(deviceExtensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &deviceExtensionCount, deviceExtensionsAvailable.data());
    std::vector<const char*> deviceExtensions;
    const bool hasPortabilitySubset = std::any_of(deviceExtensionsAvailable.begin(), deviceExtensionsAvailable.end(), [](const auto& ext) {
        return std::string(ext.extensionName) == "VK_KHR_portability_subset";
    });
    if (hasPortabilitySubset) {
        deviceExtensions.push_back("VK_KHR_portability_subset");
    }

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
    check(vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);
}

VulkanContext::~VulkanContext() {
    if (device_) vkDestroyDevice(device_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

std::vector<char> readBinaryFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Could not open binary file: " + path);
    const auto size = in.tellg();
    std::vector<char> data(static_cast<size_t>(size));
    in.seekg(0);
    in.read(data.data(), size);
    return data;
}

} // namespace sph
