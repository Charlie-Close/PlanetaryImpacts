#include <filesystem>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include "sph/Aneos.hpp"
#include "sph/Hdf5.hpp"
#include "sph/Parameters.hpp"
#include "sph/Simulator.hpp"
#include "sph/Viewer.hpp"
#include "sph/VulkanContext.hpp"
#include "sph/VulkanSimulation.hpp"

namespace {

class PhaseTimer {
public:
    explicit PhaseTimer(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {
        std::cout << name_ << "..." << std::endl;
    }
    ~PhaseTimer() {
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start_;
        std::cout << name_ << " done in " << elapsed.count() << "s" << std::endl;
    }
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

std::filesystem::path projectRoot() {
#ifdef SPH_PROJECT_ROOT
    return std::filesystem::path(SPH_PROJECT_ROOT);
#else
    return std::filesystem::current_path().parent_path();
#endif
}

std::string shellQuote(const std::string& value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

int videoFps() {
    if (const char* env = std::getenv("SPH_VIDEO_FPS")) return std::max(1, std::atoi(env));
    return 30;
}

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --input PATH          HDF5 input. Defaults to Metal FILEPATH, then demo_impact_n50.hdf5 if present.\n"
        << "  --output PATH         HDF5 output path for final state.\n"
        << "  --snapshot-dir PATH   Directory for PNG snapshots.\n"
        << "  --steps N            Number of simulation steps. Default: 1.\n"
        << "  --max-particles N    Run only the first N particles, useful for quick compatibility tests.\n"
        << "  --no-snapshot        Do not render a PNG snapshot.\n"
        << "  --no-save            Do not write final HDF5 state.\n"
        << "  --headless           Run the Vulkan GPU simulation without opening a window.\n"
        << "  --window             Open the interactive viewer. This is the default.\n"
        << "  --window-size W H    Viewer framebuffer size. Default: 1024 1024.\n"
        << "  --viewer-frames N    Render N viewer frames and exit. 0 means run until closed.\n"
        << "  --validate           Load Vulkan, EOS tables and HDF5, then exit.\n"
        << "  --require-vulkan     Fail if Vulkan cannot start.\n"
        << "  --info               Print Vulkan device information and exit.\n";
}

sph::RunOptions parseOptions(int argc, char** argv) {
    const auto root = projectRoot();
    sph::RunOptions options;
    options.input = root / sph::params::defaultInput;
    if (!std::filesystem::exists(options.input)) options.input = root / sph::params::fallbackInput;
    options.output = root / "sph_vulkan_output/state_final.hdf5";
    options.snapshotDir = root / "sph_vulkan_output/snapshots";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
            return argv[++i];
        };
        if (arg == "--input") options.input = needValue("--input");
        else if (arg == "--output") options.output = needValue("--output");
        else if (arg == "--snapshot-dir") options.snapshotDir = needValue("--snapshot-dir");
        else if (arg == "--steps") options.steps = std::stoi(needValue("--steps"));
        else if (arg == "--max-particles") options.maxParticles = std::stoi(needValue("--max-particles"));
        else if (arg == "--no-snapshot") options.snapshots = false;
        else if (arg == "--no-save") options.saveState = false;
        else if (arg == "--headless") options.headless = true;
        else if (arg == "--window") options.headless = false;
        else if (arg == "--window-size") {
            options.windowWidth = std::stoi(needValue("--window-size"));
            options.windowHeight = std::stoi(needValue("--window-size"));
        }
        else if (arg == "--viewer-frames") options.viewerFrames = std::stoi(needValue("--viewer-frames"));
        else if (arg == "--validate") options.validateOnly = true;
        else if (arg == "--require-vulkan") options.requireVulkan = true;
        else if (arg == "--info") {
            try {
                sph::VulkanContext context;
                std::cout << "Vulkan device: " << context.deviceName() << "\n";
                std::cout << "Compute queue family: " << context.computeQueueFamily() << "\n";
            } catch (const std::exception& e) {
                std::cout << "Vulkan unavailable in this runtime: " << e.what() << "\n";
                std::exit(1);
            }
            std::exit(0);
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        sph::RunOptions options = parseOptions(argc, argv);
        try {
            sph::VulkanContext context;
            std::cout << "Vulkan device: " << context.deviceName() << "\n";
        } catch (const std::exception& e) {
            if (options.requireVulkan) throw;
            std::cout << "Vulkan unavailable in this runtime: " << e.what() << "\n";
            std::cout << "Validation may continue, but simulation requires Vulkan GPU execution.\n";
        }
        std::cout << "Input: " << options.input << "\n";

        const auto root = projectRoot();
        sph::DataSet data;
        {
            PhaseTimer timer("Reading HDF5");
            data = sph::readHdf5(options.input);
        }
        std::cout << "Particles in file: " << data.positions.size() << "\n";
        if (options.maxParticles > 0) std::cout << "Particle cap: " << options.maxParticles << "\n";

        sph::EquationOfState eos;
        {
            PhaseTimer timer("Loading EOS tables");
            eos = sph::EquationOfState::load(root);
        }
        sph::Simulator simulator(std::move(data), std::move(eos), options.maxParticles);
        {
            PhaseTimer timer("Initialising simulation buffers");
            simulator.initialise();
        }
        std::cout << "Particles in run: " << simulator.particleCount() << "\n";

        if (options.validateOnly) {
            std::cout << "Validation complete.\n";
            return 0;
        }

#if SPH_HAS_VIEWER
        if (options.headless) {
            sph::VulkanSimulation gpuSimulation(simulator);
            std::cout << "Simulation Vulkan device: " << gpuSimulation.deviceName() << "\n";
            const bool quietSteps = std::getenv("SPH_QUIET_STEPS") != nullptr;
            constexpr double fpsSmoothingAlpha = 0.05;
            double smoothedFps = 0.0;
            std::array<int, sph::params::nSnapshotters> videoFrameCounts{};
            if (options.snapshots) {
                for (int snapshotter = 0; snapshotter < sph::params::nSnapshotters; ++snapshotter) {
                    const auto folder = options.snapshotDir / ("snapshotter_" + std::to_string(snapshotter));
                    std::filesystem::create_directories(folder);
                    std::filesystem::remove(folder / "frames.rgba");
                }
            }
            int nextSnapshot = 0;
            for (int i = 0; i < options.steps; ++i) {
                const auto loopStart = std::chrono::steady_clock::now();
                const auto simStart = loopStart;
                gpuSimulation.step();
                const std::chrono::duration<double> simElapsed = std::chrono::steady_clock::now() - simStart;
                if (options.snapshots && simulator.time() > static_cast<float>(nextSnapshot)) {
                    const int initialSnapshot = static_cast<int>(std::round(sph::params::startSnapshot * (1000.0f / sph::params::snapshotPeriodSeconds)));
                    const int currentFrame = static_cast<int>(std::round(static_cast<float>(nextSnapshot) / sph::params::snapshotPeriodSeconds));
                    for (int snapshotter = 0; snapshotter < sph::params::nSnapshotters; ++snapshotter) {
                        const auto folder = options.snapshotDir / ("snapshotter_" + std::to_string(snapshotter));
                        (void)initialSnapshot;
                        (void)currentFrame;
                        gpuSimulation.writeSnapshot(folder / "frames.rgba",
                                                    sph::params::snapshotPositions[snapshotter],
                                                    sph::params::snapshotPitches[snapshotter],
                                                    sph::params::snapshotYaws[snapshotter]);
                        ++videoFrameCounts[snapshotter];
                    }
                    if (options.saveState && (nextSnapshot % 1000) == 0 && simulator.time() > 500.0f) {
                        gpuSimulation.syncToSimulator();
                        const auto savePath = options.output.parent_path() /
                            ("state_" + std::to_string(sph::params::startSnapshot + ((nextSnapshot + sph::params::snapshotPeriodSeconds) / 1000)) + ".hdf5");
                        simulator.saveState(savePath);
                    }
                    nextSnapshot += sph::params::snapshotPeriodSeconds;
                }
                const std::chrono::duration<double> loopElapsed = std::chrono::steady_clock::now() - loopStart;
                const double simSeconds = simElapsed.count();
                const double loopSeconds = loopElapsed.count();
                const double fps = loopSeconds > 0.0 ? 1.0 / loopSeconds : 0.0;
                smoothedFps = smoothedFps == 0.0 ? fps : (fpsSmoothingAlpha * fps + (1.0 - fpsSmoothingAlpha) * smoothedFps);
                if (!quietSteps) {
                    std::cout << "GPU step " << (i + 1) << "/" << options.steps
                              << " time=" << simulator.time() << "s"
                              << " sim_wall=" << simSeconds << "s"
                              << " loop_wall=" << loopSeconds << "s"
                              << " fps=" << fps << " smooth_fps=" << smoothedFps << "\n";
                }
            }
            if (options.snapshots) {
                const int fps = videoFps();
                const bool keepRawFrames = std::getenv("SPH_KEEP_RAW_FRAMES") != nullptr;
                for (int snapshotter = 0; snapshotter < sph::params::nSnapshotters; ++snapshotter) {
                    if (videoFrameCounts[snapshotter] == 0) continue;
                    const auto folder = options.snapshotDir / ("snapshotter_" + std::to_string(snapshotter));
                    const auto rawPath = folder / "frames.rgba";
                    const auto videoPath = folder / ("snapshotter_" + std::to_string(snapshotter) + ".mp4");
                    const std::string command = "ffmpeg -y -hide_banner -loglevel error -f rawvideo -pixel_format rgba -video_size " +
                        std::to_string(sph::params::snapshotResolution) + "x" + std::to_string(sph::params::snapshotResolution) +
                        " -framerate " + std::to_string(fps) + " -i " + shellQuote(rawPath.string()) +
                        " -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p " + shellQuote(videoPath.string());
                    std::cout << "Encoding " << videoPath << " from " << videoFrameCounts[snapshotter] << " frames..." << std::endl;
                    if (std::system(command.c_str()) != 0) throw std::runtime_error("ffmpeg video encode failed: " + videoPath.string());
                    if (!keepRawFrames) std::filesystem::remove(rawPath);
                }
            }
            gpuSimulation.syncToSimulator();
            if (options.saveState && !options.output.empty()) simulator.saveState(options.output);
        } else {
            std::cout << "Opening interactive viewer. WASD/QE moves, drag rotates, P prints camera.\n";
            sph::runViewer(simulator, options);
        }
#else
        sph::VulkanSimulation gpuSimulation(simulator);
        std::cout << "Simulation Vulkan device: " << gpuSimulation.deviceName() << "\n";
        for (int i = 0; i < options.steps; ++i) gpuSimulation.step();
        gpuSimulation.syncToSimulator();
        if (options.saveState && !options.output.empty()) simulator.saveState(options.output);
        if (options.snapshots) {
            std::filesystem::create_directories(options.snapshotDir);
            gpuSimulation.writeSnapshot(options.snapshotDir / "snapshot_0.png", sph::params::snapshotPositions[0], sph::params::snapshotPitches[0], sph::params::snapshotYaws[0]);
        }
#endif
        std::cout << "Done.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sph_vulkan error: " << e.what() << "\n";
        return 1;
    }
}
