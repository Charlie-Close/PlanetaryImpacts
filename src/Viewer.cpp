#include "sph/Viewer.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "sph/Parameters.hpp"
#include "sph/VulkanContext.hpp"
#include "sph/VulkanSimulation.hpp"

#if defined(__APPLE__) || defined(__unix__)
#include <sys/resource.h>
#endif

namespace sph {
namespace {

constexpr int maxFramesInFlight = 2;

double peakResidentMb() {
#if defined(__APPLE__) || defined(__unix__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
#else
    return 0.0;
#endif
}

struct CameraUniform {
    std::array<float, 16> viewProj{};
    std::array<float, 16> lightViewProj{};
    float cameraPos[4]{};
    float params[4]{};
};

struct ViewerState {
    Simulator* simulator = nullptr;
    RunOptions options;
    bool paused = false;
    bool stepOnce = false;
    bool dragging = false;
    bool keys[6] = {false, false, false, false, false, false};
    bool particleDataDirty = true;
    double lastX = 0.0;
    double lastY = 0.0;
    float yaw = params::startingYaw;
    float pitch = params::startingPitch;
    Vec3 position = params::startingPosition;
    Vec3 forward = makeVec3(0.0f, 0.0f, 1.0f);
    Vec3 up = makeVec3(0.0f, 1.0f, 0.0f);
};

struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool complete() const { return graphics.has_value() && present.has_value(); }
};

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

void vkCheck(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed");
}

void updateForward(ViewerState& state) {
    const float yaw = state.yaw * 3.14159265358979323846f / 180.0f;
    const float pitch = state.pitch * 3.14159265358979323846f / 180.0f;
    state.forward = normalize(makeVec3(std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)));
}

std::array<float, 16> multiply(const std::array<float, 16>& a, const std::array<float, 16>& b) {
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

std::array<float, 16> perspective(float fovRadians, float aspect, float znear, float zfar) {
    const float ys = 1.0f / std::tan(fovRadians * 0.5f);
    const float xs = ys / aspect;
    const float zs = zfar / (znear - zfar);
    return {xs, 0.0f, 0.0f, 0.0f, 0.0f, ys, 0.0f, 0.0f, 0.0f, 0.0f, zs, -1.0f, 0.0f, 0.0f, znear * zs, 0.0f};
}

std::array<float, 16> orthographic(float left, float right, float bottom, float top, float znear, float zfar) {
    return {
        2.0f / (right - left), 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f / (top - bottom), 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f / (znear - zfar), 0.0f,
        -(right + left) / (right - left), -(top + bottom) / (top - bottom), znear / (znear - zfar), 1.0f,
    };
}

std::array<float, 16> lookAt(Vec3 pos, Vec3 forward, Vec3 up) {
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
    return multiply(rotation, translation);
}

void moveCamera(ViewerState& state) {
    constexpr float cameraSpeed = 4.0f;
    const Vec3 right = normalize(cross(state.forward, state.up));
    if (state.keys[0]) state.position += state.forward * cameraSpeed;
    if (state.keys[1]) state.position -= state.forward * cameraSpeed;
    if (state.keys[2]) state.position -= right * cameraSpeed;
    if (state.keys[3]) state.position += right * cameraSpeed;
    if (state.keys[4]) state.position += state.up * cameraSpeed;
    if (state.keys[5]) state.position -= state.up * cameraSpeed;
}

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
    auto* state = static_cast<ViewerState*>(glfwGetWindowUserPointer(window));
    const bool down = action != GLFW_RELEASE;
    if (key == GLFW_KEY_W) state->keys[0] = down;
    if (key == GLFW_KEY_S) state->keys[1] = down;
    if (key == GLFW_KEY_A) state->keys[2] = down;
    if (key == GLFW_KEY_D) state->keys[3] = down;
    if (key == GLFW_KEY_Q) state->keys[4] = down;
    if (key == GLFW_KEY_E) state->keys[5] = down;
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_P) {
        std::cout << "POSITION: " << state->position.x << ", " << state->position.y << ", " << state->position.z << "\n";
        std::cout << "PITCH, YAW: " << state->pitch << ", " << state->yaw << "\n";
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
    auto* state = static_cast<ViewerState*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        state->dragging = action == GLFW_PRESS;
        glfwGetCursorPos(window, &state->lastX, &state->lastY);
    }
}

void cursorCallback(GLFWwindow* window, double x, double y) {
    auto* state = static_cast<ViewerState*>(glfwGetWindowUserPointer(window));
    if (!state->dragging) return;
    state->yaw += static_cast<float>((x - state->lastX) * 0.1);
    state->pitch = std::clamp(state->pitch + static_cast<float>((y - state->lastY) * 0.1), -89.0f, 89.0f);
    state->lastX = x;
    state->lastY = y;
    updateForward(*state);
}

class VulkanViewer {
public:
    VulkanViewer(Simulator& simulator, const RunOptions& options) : simulator_(simulator), options_(options) {
        auto time = [](const char* name, auto&& fn) {
            const auto start = std::chrono::steady_clock::now();
            std::cout << name << "..." << std::endl;
            fn();
            const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
            std::cout << name << " done in " << elapsed.count() << "s" << std::endl;
        };
        time("Creating viewer window", [&] { initWindow(); });
        time("Creating Vulkan viewer resources", [&] { initVulkan(); });
    }
    ~VulkanViewer() { cleanup(); }
    void run();

private:
    GLFWwindow* window_ = nullptr;
    Simulator& simulator_;
    RunOptions options_;
    ViewerState state_{};
    std::unique_ptr<VulkanSimulation> gpuSimulation_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    QueueFamilies queueFamilies_;
    std::string deviceName_;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;
    VkImage shadowImage_ = VK_NULL_HANDLE;
    VkDeviceMemory shadowMemory_ = VK_NULL_HANDLE;
    VkImageView shadowImageView_ = VK_NULL_HANDLE;
    VkSampler shadowSampler_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_D16_UNORM;
    VkFormat shadowDepthFormat_ = VK_FORMAT_D32_SFLOAT;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass_ = VK_NULL_HANDLE;
    VkFramebuffer shadowFramebuffer_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout packDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout packPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline packPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool packDescriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet packDescriptorSet_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    Buffer quadBuffer_;
    Buffer indexBuffer_;
    Buffer instanceIdBuffer_;
    Buffer visibleCountBuffer_;
    Buffer indirectDrawBuffer_;
    std::vector<Buffer> uniformBuffers_;
    uint32_t particleCount_ = 0;

    std::vector<VkSemaphore> imageAvailable_;
    std::vector<VkSemaphore> renderFinished_;
    std::vector<VkFence> inFlight_;
    size_t currentFrame_ = 0;
    int renderedFrames_ = 0;

    void initWindow();
    void initVulkan();
    void cleanup();
    void createInstance();
    void pickPhysicalDevice();
    void createDevice();
    void createSwapchain();
    void createImageViews();
    void createDepthResources();
    void createRenderPass();
    void createShadowResources();
    void createDescriptorSetLayout();
    void createPipeline();
    void createShadowPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createBuffers();
    void createDescriptorPoolAndSets();
    void createPackResources();
    void createCommandBuffers();
    void createSync();
    void drawFrame();
    void recordPackParticles(VkCommandBuffer cmd);
    void recordShadowPass(VkCommandBuffer cmd);
    void updateUniform(uint32_t frameIndex);
    QueueFamilies findQueueFamilies(VkPhysicalDevice device);
    uint32_t findMemory(uint32_t typeBits, VkMemoryPropertyFlags flags);
    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory);
    VkShaderModule shaderModule(const std::string& path);
};

void VulkanViewer::initWindow() {
    if (!glfwInit()) throw std::runtime_error("Could not initialize GLFW");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(options_.windowWidth, options_.windowHeight, "Fluid Simulation", nullptr, nullptr);
    if (!window_) throw std::runtime_error("Could not create GLFW window");
    state_.simulator = &simulator_;
    state_.options = options_;
    updateForward(state_);
    glfwSetWindowUserPointer(window_, &state_);
    glfwSetKeyCallback(window_, keyCallback);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetCursorPosCallback(window_, cursorCallback);
}

void VulkanViewer::createInstance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "sph_vulkan";
    app.apiVersion = VK_API_VERSION_1_1;
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    vkCheck(vkCreateInstance(&info, nullptr, &instance_), "vkCreateInstance");
}

QueueFamilies VulkanViewer::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilies out;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
    for (uint32_t i = 0; i < count; ++i) {
        if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) out.graphics = i;
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present);
        if (present) out.present = i;
        if (out.complete()) break;
    }
    return out;
}

void VulkanViewer::pickPhysicalDevice() {
    uint32_t count = 0;
    vkCheck(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devices(count);
    vkCheck(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");
    for (VkPhysicalDevice device : devices) {
        QueueFamilies families = findQueueFamilies(device);
        if (!families.complete()) continue;
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, exts.data());
        bool hasSwapchain = false;
        for (const auto& ext : exts) hasSwapchain = hasSwapchain || std::strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
        if (!hasSwapchain) continue;
        physicalDevice_ = device;
        queueFamilies_ = families;
        VkPhysicalDeviceProperties deviceProps{};
        vkGetPhysicalDeviceProperties(device, &deviceProps);
        deviceName_ = deviceProps.deviceName;
        std::cout << "Viewer Vulkan device: " << deviceName_ << "\n";
        return;
    }
    throw std::runtime_error("No suitable Vulkan graphics/present device found");
}

void VulkanViewer::createDevice() {
    std::set<uint32_t> unique = {*queueFamilies_.graphics, *queueFamilies_.present};
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queues;
    for (uint32_t family : unique) {
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        queues.push_back(queue);
    }
    std::vector<const char*> extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset"};
    VkPhysicalDeviceFeatures supportedFeatures{};
    vkGetPhysicalDeviceFeatures(physicalDevice_, &supportedFeatures);
    if (!supportedFeatures.shaderStorageBufferArrayDynamicIndexing) {
        throw std::runtime_error("Vulkan device does not support shaderStorageBufferArrayDynamicIndexing");
    }
    VkPhysicalDeviceFeatures enabledFeatures{};
    enabledFeatures.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
    info.pQueueCreateInfos = queues.data();
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    info.pEnabledFeatures = &enabledFeatures;
    vkCheck(vkCreateDevice(physicalDevice_, &info, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, *queueFamilies_.graphics, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, *queueFamilies_.present, 0, &presentQueue_);
}

void VulkanViewer::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &caps);
    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, formats.data());
    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB) {
            chosen = f;
            break;
        }
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) chosen = f;
    }
    int w = 0, h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    swapchainExtent_ = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : VkExtent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    uint32_t imageCount = std::clamp(caps.minImageCount + 1, caps.minImageCount, caps.maxImageCount ? caps.maxImageCount : caps.minImageCount + 1);
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = surface_;
    info.minImageCount = imageCount;
    info.imageFormat = chosen.format;
    info.imageColorSpace = chosen.colorSpace;
    info.imageExtent = swapchainExtent_;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    uint32_t q[] = {*queueFamilies_.graphics, *queueFamilies_.present};
    if (q[0] != q[1]) {
        info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = 2;
        info.pQueueFamilyIndices = q;
    } else {
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    vkCheck(vkCreateSwapchainKHR(device_, &info, nullptr, &swapchain_), "vkCreateSwapchainKHR");
    swapchainFormat_ = chosen.format;
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
    swapchainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());
}

void VulkanViewer::createImageViews() {
    for (VkImage image : swapchainImages_) {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = swapchainFormat_;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        VkImageView view;
        vkCheck(vkCreateImageView(device_, &info, nullptr, &view), "vkCreateImageView");
        swapchainImageViews_.push_back(view);
    }
}

void VulkanViewer::createRenderPass() {
    VkAttachmentDescription color{};
    color.format = swapchainFormat_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentDescription depth{};
    depth.format = depthFormat_;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;
    subpass.pDepthStencilAttachment = &depthRef;
    std::array<VkAttachmentDescription, 2> attachments = {color, depth};
    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = static_cast<uint32_t>(attachments.size());
    info.pAttachments = attachments.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    vkCheck(vkCreateRenderPass(device_, &info, nullptr, &renderPass_), "vkCreateRenderPass");

    VkAttachmentDescription shadowDepth{};
    shadowDepth.format = shadowDepthFormat_;
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
    std::array<VkSubpassDependency, 2> shadowDeps{};
    shadowDeps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    shadowDeps[0].dstSubpass = 0;
    shadowDeps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    shadowDeps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    shadowDeps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    shadowDeps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    shadowDeps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    shadowDeps[1].srcSubpass = 0;
    shadowDeps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    shadowDeps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    shadowDeps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    shadowDeps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    shadowDeps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    shadowDeps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    VkRenderPassCreateInfo shadowInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    shadowInfo.attachmentCount = 1;
    shadowInfo.pAttachments = &shadowDepth;
    shadowInfo.subpassCount = 1;
    shadowInfo.pSubpasses = &shadowSubpass;
    shadowInfo.dependencyCount = static_cast<uint32_t>(shadowDeps.size());
    shadowInfo.pDependencies = shadowDeps.data();
    vkCheck(vkCreateRenderPass(device_, &shadowInfo, nullptr, &shadowRenderPass_), "vkCreateRenderPass shadow");
}

void VulkanViewer::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 10> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (uint32_t i = 2; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = static_cast<uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    vkCheck(vkCreateDescriptorSetLayout(device_, &info, nullptr, &descriptorSetLayout_), "vkCreateDescriptorSetLayout");
}

VkShaderModule VulkanViewer::shaderModule(const std::string& path) {
    std::vector<char> bytes = readBinaryFile(path);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytes.size();
    info.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
    VkShaderModule module;
    vkCheck(vkCreateShaderModule(device_, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

void VulkanViewer::createPipeline() {
    VkShaderModule vert = shaderModule("build/shaders/particle.vert.spv");
    VkShaderModule frag = shaderModule("build/shaders/particle.frag.spv");
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
    VkViewport viewport{0, 0, static_cast<float>(swapchainExtent_.width), static_cast<float>(swapchainExtent_.height), 0, 1};
    VkRect2D scissor{{0, 0}, swapchainExtent_};
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1; viewportState.pViewports = &viewport; viewportState.scissorCount = 1; viewportState.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL; raster.cullMode = VK_CULL_MODE_NONE; raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blendState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blendState.attachmentCount = 1; blendState.pAttachments = &blend;
    VkPipelineDepthStencilStateCreateInfo depthState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthState.depthTestEnable = VK_TRUE;
    depthState.depthWriteEnable = VK_TRUE;
    depthState.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1; layout.pSetLayouts = &descriptorSetLayout_;
    vkCheck(vkCreatePipelineLayout(device_, &layout, nullptr, &pipelineLayout_), "vkCreatePipelineLayout");
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = 2; info.pStages = stages; info.pVertexInputState = &vertexInput; info.pInputAssemblyState = &assembly; info.pViewportState = &viewportState;
    info.pRasterizationState = &raster; info.pMultisampleState = &msaa; info.pDepthStencilState = &depthState; info.pColorBlendState = &blendState; info.layout = pipelineLayout_; info.renderPass = renderPass_;
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_), "vkCreateGraphicsPipelines");
    vkDestroyShaderModule(device_, vert, nullptr); vkDestroyShaderModule(device_, frag, nullptr);
}

void VulkanViewer::createShadowPipeline() {
    VkShaderModule vert = shaderModule("build/shaders/particle_shadow.vert.spv");
    VkShaderModule frag = shaderModule("build/shaders/particle_shadow.frag.spv");
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
    VkViewport viewport{0, 0, 1024.0f, 1024.0f, 0, 1};
    VkRect2D scissor{{0, 0}, {1024, 1024}};
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
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount = 1; layout.pSetLayouts = &descriptorSetLayout_;
    vkCheck(vkCreatePipelineLayout(device_, &layout, nullptr, &shadowPipelineLayout_), "vkCreatePipelineLayout shadow");
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = 2; info.pStages = stages; info.pVertexInputState = &vertexInput; info.pInputAssemblyState = &assembly; info.pViewportState = &viewportState;
    info.pRasterizationState = &raster; info.pMultisampleState = &msaa; info.pDepthStencilState = &depthState; info.layout = shadowPipelineLayout_; info.renderPass = shadowRenderPass_;
    vkCheck(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &shadowPipeline_), "vkCreateGraphicsPipelines shadow");
    vkDestroyShaderModule(device_, vert, nullptr); vkDestroyShaderModule(device_, frag, nullptr);
}

void VulkanViewer::createFramebuffers() {
    for (VkImageView view : swapchainImageViews_) {
        std::array<VkImageView, 2> attachments = {view, depthImageView_};
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = renderPass_;
        info.attachmentCount = static_cast<uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.width = swapchainExtent_.width;
        info.height = swapchainExtent_.height;
        info.layers = 1;
        VkFramebuffer fb;
        vkCheck(vkCreateFramebuffer(device_, &info, nullptr, &fb), "vkCreateFramebuffer");
        framebuffers_.push_back(fb);
    }
}

uint32_t VulkanViewer::findMemory(uint32_t typeBits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("No matching Vulkan memory type");
}

Buffer VulkanViewer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    Buffer out; out.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size; info.usage = usage; info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateBuffer(device_, &info, nullptr, &out.buffer), "vkCreateBuffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, out.buffer, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size; alloc.memoryTypeIndex = findMemory(req.memoryTypeBits, properties);
    vkCheck(vkAllocateMemory(device_, &alloc, nullptr, &out.memory), "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(device_, out.buffer, out.memory, 0), "vkBindBufferMemory");
    return out;
}

void VulkanViewer::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.format = format;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage = usage;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCheck(vkCreateImage(device_, &info, nullptr, &image), "vkCreateImage");
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, image, &req);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = findMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCheck(vkAllocateMemory(device_, &alloc, nullptr, &memory), "vkAllocateMemory");
    vkCheck(vkBindImageMemory(device_, image, memory, 0), "vkBindImageMemory");
}

void VulkanViewer::createDepthResources() {
    createImage(swapchainExtent_.width, swapchainExtent_.height, depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImage_, depthMemory_);
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = depthImage_;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = depthFormat_;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.layerCount = 1;
    vkCheck(vkCreateImageView(device_, &info, nullptr, &depthImageView_), "vkCreateImageView");
}

void VulkanViewer::createShadowResources() {
    constexpr uint32_t shadowSize = 1024;
    createImage(shadowSize, shadowSize, shadowDepthFormat_,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                shadowImage_, shadowMemory_);
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = shadowImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = shadowDepthFormat_;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    vkCheck(vkCreateImageView(device_, &viewInfo, nullptr, &shadowImageView_), "vkCreateImageView shadow");

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = 1.0f;
    sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    vkCheck(vkCreateSampler(device_, &sampler, nullptr, &shadowSampler_), "vkCreateSampler shadow");

    VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb.renderPass = shadowRenderPass_;
    fb.attachmentCount = 1;
    fb.pAttachments = &shadowImageView_;
    fb.width = shadowSize;
    fb.height = shadowSize;
    fb.layers = 1;
    vkCheck(vkCreateFramebuffer(device_, &fb, nullptr, &shadowFramebuffer_), "vkCreateFramebuffer shadow");
}

void VulkanViewer::createCommandPool() {
    VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.queueFamilyIndex = *queueFamilies_.graphics;
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCheck(vkCreateCommandPool(device_, &info, nullptr, &commandPool_), "vkCreateCommandPool");
}

void VulkanViewer::createBuffers() {
    std::array<float, 8> quad = {-1, -1, 1, -1, 1, 1, -1, 1};
    quadBuffer_ = createBuffer(sizeof(quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped = nullptr;
    vkMapMemory(device_, quadBuffer_.memory, 0, sizeof(quad), 0, &mapped);
    std::memcpy(mapped, quad.data(), sizeof(quad));
    vkUnmapMemory(device_, quadBuffer_.memory);
    std::array<uint16_t, 6> indices = {0, 1, 2, 0, 2, 3};
    indexBuffer_ = createBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkMapMemory(device_, indexBuffer_.memory, 0, sizeof(indices), 0, &mapped);
    std::memcpy(mapped, indices.data(), sizeof(indices));
    vkUnmapMemory(device_, indexBuffer_.memory);
    particleCount_ = gpuSimulation_->particleCount();
    instanceIdBuffer_ = createBuffer(static_cast<VkDeviceSize>(particleCount_) * sizeof(uint32_t),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    visibleCountBuffer_ = createBuffer(sizeof(uint32_t),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    indirectDrawBuffer_ = createBuffer(5 * sizeof(uint32_t),
                                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::cout << "Renderable particles: " << particleCount_ << " / " << simulator_.data().positions.size() << std::endl;
    for (int i = 0; i < maxFramesInFlight; ++i) {
        uniformBuffers_.push_back(createBuffer(sizeof(CameraUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
    }
}

void VulkanViewer::createDescriptorPoolAndSets() {
    std::array<VkDescriptorPoolSize, 3> pools{};
    pools[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxFramesInFlight};
    pools[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxFramesInFlight};
    pools[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxFramesInFlight * 8};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = maxFramesInFlight;
    poolInfo.poolSizeCount = static_cast<uint32_t>(pools.size());
    poolInfo.pPoolSizes = pools.data();
    vkCheck(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_), "vkCreateDescriptorPool");
    std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout_);
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = descriptorPool_; alloc.descriptorSetCount = maxFramesInFlight; alloc.pSetLayouts = layouts.data();
    descriptorSets_.resize(maxFramesInFlight);
    vkCheck(vkAllocateDescriptorSets(device_, &alloc, descriptorSets_.data()), "vkAllocateDescriptorSets");
    for (int i = 0; i < maxFramesInFlight; ++i) {
        VkDescriptorBufferInfo buffer{uniformBuffers_[i].buffer, 0, sizeof(CameraUniform)};
        VkDescriptorImageInfo shadow{shadowSampler_, shadowImageView_, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        std::array<VkDescriptorBufferInfo, 8> storage{};
        storage[0] = {gpuSimulation_->positionsBuffer().buffer, 0, gpuSimulation_->positionsBuffer().size};
        storage[1] = {gpuSimulation_->densitiesBuffer().buffer, 0, gpuSimulation_->densitiesBuffer().size};
        storage[2] = {gpuSimulation_->smoothingLengthsBuffer().buffer, 0, gpuSimulation_->smoothingLengthsBuffer().size};
        storage[3] = {gpuSimulation_->temperaturesBuffer().buffer, 0, gpuSimulation_->temperaturesBuffer().size};
        storage[4] = {gpuSimulation_->materialIdsBuffer().buffer, 0, gpuSimulation_->materialIdsBuffer().size};
        storage[5] = {gpuSimulation_->rhoGradsBuffer().buffer, 0, gpuSimulation_->rhoGradsBuffer().size};
        storage[6] = {gpuSimulation_->alphaBuffer().buffer, 0, gpuSimulation_->alphaBuffer().size};
        storage[7] = {instanceIdBuffer_.buffer, 0, instanceIdBuffer_.size};
        std::array<VkWriteDescriptorSet, 10> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &buffer;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets_[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &shadow;
        for (uint32_t j = 0; j < storage.size(); ++j) {
            writes[2 + j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2 + j].dstSet = descriptorSets_[i];
            writes[2 + j].dstBinding = 2 + j;
            writes[2 + j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[2 + j].descriptorCount = 1;
            writes[2 + j].pBufferInfo = &storage[j];
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void VulkanViewer::createPackResources() {
    std::array<VkDescriptorSetLayoutBinding, 10> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkCheck(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &packDescriptorSetLayout_), "vkCreateDescriptorSetLayout");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = 16;
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &packDescriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push;
    vkCheck(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &packPipelineLayout_), "vkCreatePipelineLayout");

    VkShaderModule shader = shaderModule("build/shaders/pack_particles.comp.spv");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = stage;
    pipelineInfo.layout = packPipelineLayout_;
    vkCheck(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &packPipeline_), "vkCreateComputePipelines");
    vkDestroyShaderModule(device_, shader, nullptr);

    VkDescriptorPoolSize pool{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(bindings.size())};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &pool;
    vkCheck(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &packDescriptorPool_), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = packDescriptorPool_;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &packDescriptorSetLayout_;
    vkCheck(vkAllocateDescriptorSets(device_, &alloc, &packDescriptorSet_), "vkAllocateDescriptorSets");

    std::array<VkDescriptorBufferInfo, 10> infos{};
    infos[0] = {gpuSimulation_->positionsBuffer().buffer, 0, gpuSimulation_->positionsBuffer().size};
    infos[1] = {gpuSimulation_->densitiesBuffer().buffer, 0, gpuSimulation_->densitiesBuffer().size};
    infos[2] = {gpuSimulation_->smoothingLengthsBuffer().buffer, 0, gpuSimulation_->smoothingLengthsBuffer().size};
    infos[3] = {gpuSimulation_->temperaturesBuffer().buffer, 0, gpuSimulation_->temperaturesBuffer().size};
    infos[4] = {gpuSimulation_->materialIdsBuffer().buffer, 0, gpuSimulation_->materialIdsBuffer().size};
    infos[5] = {gpuSimulation_->rhoGradsBuffer().buffer, 0, gpuSimulation_->rhoGradsBuffer().size};
    infos[6] = {gpuSimulation_->alphaBuffer().buffer, 0, gpuSimulation_->alphaBuffer().size};
    infos[7] = {instanceIdBuffer_.buffer, 0, instanceIdBuffer_.size};
    infos[8] = {visibleCountBuffer_.buffer, 0, visibleCountBuffer_.size};
    infos[9] = {indirectDrawBuffer_.buffer, 0, indirectDrawBuffer_.size};
    std::array<VkWriteDescriptorSet, 10> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = packDescriptorSet_;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void VulkanViewer::createCommandBuffers() {
    commandBuffers_.resize(maxFramesInFlight);
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = commandPool_; alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
    vkCheck(vkAllocateCommandBuffers(device_, &alloc, commandBuffers_.data()), "vkAllocateCommandBuffers");
}

void VulkanViewer::createSync() {
    VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    imageAvailable_.resize(maxFramesInFlight); renderFinished_.resize(maxFramesInFlight); inFlight_.resize(maxFramesInFlight);
    for (int i = 0; i < maxFramesInFlight; ++i) {
        vkCheck(vkCreateSemaphore(device_, &sem, nullptr, &imageAvailable_[i]), "vkCreateSemaphore");
        vkCheck(vkCreateSemaphore(device_, &sem, nullptr, &renderFinished_[i]), "vkCreateSemaphore");
        vkCheck(vkCreateFence(device_, &fence, nullptr, &inFlight_[i]), "vkCreateFence");
    }
}

void VulkanViewer::updateUniform(uint32_t frameIndex) {
    int w = 0, h = 0; glfwGetFramebufferSize(window_, &w, &h);
    CameraUniform u{};
    const auto proj = perspective(45.0f * 3.14159265358979323846f / 180.0f, static_cast<float>(std::max(w, 1)) / std::max(h, 1), 10.0f, 1000.0f);
    const auto view = lookAt(state_.position, state_.forward, state_.up);
    u.viewProj = multiply(proj, view);
    const Vec3 lightDir = normalize(params::lightDirection);
    const Vec3 lightPos = makeVec3(params::boxCenter, params::boxCenter, params::boxCenter) - lightDir * 300.0f;
    const auto lightView = lookAt(lightPos, lightDir, makeVec3(0.0f, 1.0f, 0.0f));
    const auto lightProj = orthographic(-50.0f, 50.0f, -50.0f, 50.0f, 0.0f, 500.0f);
    u.lightViewProj = multiply(lightProj, lightView);
    u.cameraPos[0] = state_.position.x; u.cameraPos[1] = state_.position.y; u.cameraPos[2] = state_.position.z;
    u.params[0] = params::particleSize;
    void* mapped = nullptr;
    vkMapMemory(device_, uniformBuffers_[frameIndex].memory, 0, sizeof(u), 0, &mapped);
    std::memcpy(mapped, &u, sizeof(u));
    vkUnmapMemory(device_, uniformBuffers_[frameIndex].memory);
}

void VulkanViewer::recordPackParticles(VkCommandBuffer cmd) {
    struct PackPush {
        int32_t count = 0;
        float cameraX = 0.0f;
        float cameraY = 0.0f;
        float cameraZ = 0.0f;
    } push{static_cast<int32_t>(particleCount_), state_.position.x, state_.position.y, state_.position.z};
    const uint32_t drawArgs[5] = {6u, 0u, 0u, 0u, 0u};
    vkCmdFillBuffer(cmd, visibleCountBuffer_.buffer, 0, visibleCountBuffer_.size, 0u);
    vkCmdUpdateBuffer(cmd, indirectDrawBuffer_.buffer, 0, sizeof(drawArgs), drawArgs);
    VkMemoryBarrier clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    clearBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &clearBarrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packPipelineLayout_, 0, 1, &packDescriptorSet_, 0, nullptr);
    vkCmdPushConstants(cmd, packPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (particleCount_ + 255u) / 256u, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0,
                         1, &barrier, 0, nullptr, 0, nullptr);
    VkBufferCopy copy{};
    copy.srcOffset = 0;
    copy.dstOffset = sizeof(uint32_t);
    copy.size = sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, visibleCountBuffer_.buffer, indirectDrawBuffer_.buffer, 1, &copy);
    VkMemoryBarrier drawBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    drawBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    drawBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0,
                         1, &drawBarrier, 0, nullptr, 0, nullptr);
}

void VulkanViewer::recordShadowPass(VkCommandBuffer cmd) {
    VkClearValue clear{};
    clear.depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = shadowRenderPass_;
    rp.framebuffer = shadowFramebuffer_;
    rp.renderArea.extent = {1024, 1024};
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &quadBuffer_.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuffer_.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipelineLayout_, 0, 1, &descriptorSets_[currentFrame_], 0, nullptr);
    vkCmdDrawIndexedIndirect(cmd, indirectDrawBuffer_.buffer, 0, 1, 0);
    vkCmdEndRenderPass(cmd);
}

void VulkanViewer::drawFrame() {
    vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);
    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
    if (acquire != VK_SUCCESS) return;
    vkResetFences(device_, 1, &inFlight_[currentFrame_]);
    updateUniform(static_cast<uint32_t>(currentFrame_));
    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin);
    recordPackParticles(cmd);
    recordShadowPass(cmd);
    std::array<VkClearValue, 2> clear{};
    clear[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clear[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = renderPass_; rp.framebuffer = framebuffers_[imageIndex]; rp.renderArea.extent = swapchainExtent_; rp.clearValueCount = static_cast<uint32_t>(clear.size()); rp.pClearValues = clear.data();
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &quadBuffer_.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, indexBuffer_.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &descriptorSets_[currentFrame_], 0, nullptr);
    vkCmdDrawIndexedIndirect(cmd, indirectDrawBuffer_.buffer, 0, 1, 0);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = &imageAvailable_[currentFrame_]; submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd; submit.signalSemaphoreCount = 1; submit.pSignalSemaphores = &renderFinished_[currentFrame_];
    vkCheck(vkQueueSubmit(graphicsQueue_, 1, &submit, inFlight_[currentFrame_]), "vkQueueSubmit");
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1; present.pWaitSemaphores = &renderFinished_[currentFrame_]; present.swapchainCount = 1; present.pSwapchains = &swapchain_; present.pImageIndices = &imageIndex;
    vkQueuePresentKHR(presentQueue_, &present);
    currentFrame_ = (currentFrame_ + 1) % maxFramesInFlight;
}

void VulkanViewer::initVulkan() {
    createInstance();
    vkCheck(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_), "glfwCreateWindowSurface");
    pickPhysicalDevice();
    createDevice();
    gpuSimulation_ = std::make_unique<VulkanSimulation>(simulator_, physicalDevice_, device_, graphicsQueue_, *queueFamilies_.graphics, deviceName_);
    std::cout << "Simulation Vulkan device: " << gpuSimulation_->deviceName() << "\n";
    createSwapchain();
    createImageViews();
    createRenderPass();
    createDepthResources();
    createShadowResources();
    createDescriptorSetLayout();
    createPipeline();
    createShadowPipeline();
    createFramebuffers();
    createCommandPool();
    createBuffers();
    createPackResources();
    gpuSimulation_->step();
    createDescriptorPoolAndSets();
    createCommandBuffers();
    createSync();
}

void VulkanViewer::run() {
    const auto loopStart = std::chrono::steady_clock::now();
    auto lastTitleUpdate = loopStart;
    int titleFrames = 0;
    const bool bench = std::getenv("SPH_VIEWER_BENCH") != nullptr;
    int benchInterval = 1;
    if (const char* interval = std::getenv("SPH_VIEWER_BENCH_INTERVAL")) {
        benchInterval = std::max(1, std::atoi(interval));
    }
    while (!glfwWindowShouldClose(window_)) {
        const int frameNumber = renderedFrames_ + 1;
        const bool printBench = bench && ((frameNumber - 1) % benchInterval == 0);
        const auto frameStart = std::chrono::steady_clock::now();
        if (printBench) {
            std::cout << "[viewer] frame=" << frameNumber << " begin rss_mb=" << peakResidentMb() << std::endl;
        }
        glfwPollEvents();
        moveCamera(state_);
        const auto simStart = std::chrono::steady_clock::now();
        if (!state_.paused || state_.stepOnce) {
            gpuSimulation_->step();
            state_.stepOnce = false;
        }
        const auto simEnd = std::chrono::steady_clock::now();
        if (printBench) {
            const std::chrono::duration<double, std::milli> simElapsed = simEnd - simStart;
            std::cout << "[viewer] frame=" << frameNumber << " after_sim sim_ms=" << simElapsed.count()
                      << " sim_time=" << simulator_.time() << " rss_mb=" << peakResidentMb() << std::endl;
        }
        const auto drawStart = std::chrono::steady_clock::now();
        drawFrame();
        const auto drawEnd = std::chrono::steady_clock::now();
        if (printBench) {
            const std::chrono::duration<double, std::milli> simElapsed = simEnd - simStart;
            const std::chrono::duration<double, std::milli> drawElapsed = drawEnd - drawStart;
            const std::chrono::duration<double, std::milli> totalElapsed = drawEnd - frameStart;
            std::cout << "[viewer] frame=" << frameNumber
                      << " done sim_ms=" << simElapsed.count()
                      << " draw_ms=" << drawElapsed.count()
                      << " total_ms=" << totalElapsed.count()
                      << " rss_mb=" << peakResidentMb() << std::endl;
        }
        ++titleFrames;
        const auto titleNow = std::chrono::steady_clock::now();
        const std::chrono::duration<double> titleElapsed = titleNow - lastTitleUpdate;
        if (titleElapsed.count() >= 0.5) {
            const double fps = static_cast<double>(titleFrames) / titleElapsed.count();
            const std::string title = std::string("Fluid Simulation - ") + std::to_string(static_cast<int>(fps + 0.5)) + " fps";
            glfwSetWindowTitle(window_, title.c_str());
            titleFrames = 0;
            lastTitleUpdate = titleNow;
        }
        if (options_.viewerFrames > 0 && ++renderedFrames_ >= options_.viewerFrames) glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
    vkDeviceWaitIdle(device_);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - loopStart;
    std::cout << "Viewer loop rendered " << renderedFrames_ << " frames in " << elapsed.count() << "s";
    if (elapsed.count() > 0.0) std::cout << " (" << renderedFrames_ / elapsed.count() << " fps)";
    std::cout << std::endl;
}

void VulkanViewer::cleanup() {
    if (device_) vkDeviceWaitIdle(device_);
    gpuSimulation_.reset();
    for (auto fence : inFlight_) vkDestroyFence(device_, fence, nullptr);
    for (auto sem : imageAvailable_) vkDestroySemaphore(device_, sem, nullptr);
    for (auto sem : renderFinished_) vkDestroySemaphore(device_, sem, nullptr);
    if (packDescriptorPool_) vkDestroyDescriptorPool(device_, packDescriptorPool_, nullptr);
    if (packPipeline_) vkDestroyPipeline(device_, packPipeline_, nullptr);
    if (packPipelineLayout_) vkDestroyPipelineLayout(device_, packPipelineLayout_, nullptr);
    if (packDescriptorSetLayout_) vkDestroyDescriptorSetLayout(device_, packDescriptorSetLayout_, nullptr);
    for (auto& b : uniformBuffers_) { vkDestroyBuffer(device_, b.buffer, nullptr); vkFreeMemory(device_, b.memory, nullptr); }
    if (indirectDrawBuffer_.buffer) { vkDestroyBuffer(device_, indirectDrawBuffer_.buffer, nullptr); vkFreeMemory(device_, indirectDrawBuffer_.memory, nullptr); }
    if (visibleCountBuffer_.buffer) { vkDestroyBuffer(device_, visibleCountBuffer_.buffer, nullptr); vkFreeMemory(device_, visibleCountBuffer_.memory, nullptr); }
    if (instanceIdBuffer_.buffer) { vkDestroyBuffer(device_, instanceIdBuffer_.buffer, nullptr); vkFreeMemory(device_, instanceIdBuffer_.memory, nullptr); }
    if (indexBuffer_.buffer) { vkDestroyBuffer(device_, indexBuffer_.buffer, nullptr); vkFreeMemory(device_, indexBuffer_.memory, nullptr); }
    if (quadBuffer_.buffer) { vkDestroyBuffer(device_, quadBuffer_.buffer, nullptr); vkFreeMemory(device_, quadBuffer_.memory, nullptr); }
    if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    if (shadowPipeline_) vkDestroyPipeline(device_, shadowPipeline_, nullptr);
    if (shadowPipelineLayout_) vkDestroyPipelineLayout(device_, shadowPipelineLayout_, nullptr);
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    if (descriptorSetLayout_) vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
    if (shadowFramebuffer_) vkDestroyFramebuffer(device_, shadowFramebuffer_, nullptr);
    if (shadowRenderPass_) vkDestroyRenderPass(device_, shadowRenderPass_, nullptr);
    if (shadowSampler_) vkDestroySampler(device_, shadowSampler_, nullptr);
    if (shadowImageView_) vkDestroyImageView(device_, shadowImageView_, nullptr);
    if (shadowImage_) vkDestroyImage(device_, shadowImage_, nullptr);
    if (shadowMemory_) vkFreeMemory(device_, shadowMemory_, nullptr);
    if (renderPass_) vkDestroyRenderPass(device_, renderPass_, nullptr);
    if (depthImageView_) vkDestroyImageView(device_, depthImageView_, nullptr);
    if (depthImage_) vkDestroyImage(device_, depthImage_, nullptr);
    if (depthMemory_) vkFreeMemory(device_, depthMemory_, nullptr);
    for (auto view : swapchainImageViews_) vkDestroyImageView(device_, view, nullptr);
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    if (device_) vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

} // namespace

void runViewer(Simulator& simulator, const RunOptions& options) {
    VulkanViewer viewer(simulator, options);
    viewer.run();
}

} // namespace sph
