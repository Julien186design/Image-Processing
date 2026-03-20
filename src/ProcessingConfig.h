#ifndef IMAGE_PROCESSING_PROCESSINGCONFIG_H
#define IMAGE_PROCESSING_PROCESSINGCONFIG_H

#define THREAD_OFFSET 1

#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <array>
#include <iostream>
#include <filesystem>

struct parameters {
    static constexpr std::array<float, 3> proportions = {0.f, 1.f, 0.5f};
    static constexpr std::array<int, 3> colorNuances = {0, 80, 20}; // {first color, last color, step}
    static constexpr std::array<int, 2> frames = {0, 600};
    static constexpr int fps = 10;
    static constexpr int fraction = 3;
    static constexpr std::array<int, 2> rectangles = {40, 63};
    static constexpr std::array<int, 3> toleranceOneColor = {20, 30, 10};
    static constexpr std::array<float, 3> weightOfRGB = {0.f, 1.f, 0.5f};
    static constexpr bool complete_transformation_colors_by_proportion = false;
    static constexpr bool oneColor = true;
    static constexpr bool totalReversal = false;
    static constexpr bool partial = false;
    static constexpr bool partialInDiagonal = false;
    static constexpr int numProportionSteps =
    static_cast<int>((proportions[1] - proportions[0]) / proportions[2]) + 1;
};

struct ThresholdParams {
    bool below;
    bool dark;
};

// ─── Transformation entry: groups a suffix with its output directory ──────────

struct TransformationEntry {
    std::string suffix;
    std::string output_dir;

    // Returns the "partial/square" variant of this entry
    [[nodiscard]] TransformationEntry partial() const {
        std::string dir = output_dir;
        if (!dir.empty() && dir.back() == '/')
            dir.pop_back();
        return { suffix + " Partial", dir + " Square/" };
    }
};

// ─── Threshold parameter sets ─────────────────────────────────────────────────

constexpr std::array<ThresholdParams, 4> transformation_params = {{
    {true,  true },  // BelowDark
    {true,  false},  // BelowLight
    {false, true },  // AboveDark
    {false, false}   // AboveLight
}};


extern const std::string folder_output;
extern const std::string folder_50;
extern const std::string folder_videos;
extern const std::string folder_edgedetector;
extern const std::string folder_onecolor;


extern const std::vector<TransformationEntry> total_step_by_step_entries;
extern const std::vector<TransformationEntry> reversal_step_by_step_entries;
extern const std::vector<TransformationEntry> total_black_and_white_entries;

void image_and_video_processing(const std::string& inputFile);

inline void sendNotification(const std::string& title, const std::string &message)
{
    const std::string command =
        "notify-send '" + title + "' '" + message + "'";

    (void)std::system(command.c_str());
}

inline std::vector<int> generateDiagonalRectangles() {
    if constexpr (parameters::fraction <= 0) return {};
    constexpr int taille = 1 << parameters::fraction;  // 2^fraction via bit-shift
    constexpr int pas = taille + 1;

    std::vector<int> vector;
    vector.reserve(taille);

    vector.push_back(-1);
    for (int i = 0; i < taille; ++i) {
        vector.push_back(i * pas);
    }
    return vector;
}

inline std::pair<bool, std::vector<int>> decode_rectangles() {
    if constexpr (parameters::rectangles.empty()) return {false, {}};

    if constexpr (parameters::rectangles[0] == -1) {
        // Diagonal case: rectangles are explicitly listed after the flag
        return {true, std::vector<int>(parameters::rectangles.begin() + 1, parameters::rectangles.end())};
    }

    // Non-diagonal case: input is {start, end}, expand to range
    if constexpr (parameters::rectangles.size() != 2) return {false, {}};
    constexpr int start = std::min(parameters::rectangles[0], parameters::rectangles[1]);
    constexpr int end   = std::max(parameters::rectangles[0], parameters::rectangles[1]);

    std::vector<int> result;
    result.reserve(end - start + 1);
    for (int i = start; i <= end; ++i)
        result.push_back(i);
    return {false, result};
}



inline bool is_mp4_file(const std::string& path) {
    const auto ext = std::filesystem::path(path).extension().string();
    std::string lower = ext;
    std::ranges::transform(lower, lower.begin(),
        [](const unsigned char c) { return std::tolower(c); });
    return lower == ".mp4";
}

inline int computeNumThreads() {
    return std::max(1, omp_get_max_threads() - THREAD_OFFSET);
}


// ─── Partial variant generation ───────────────────────────────────────────────

inline std::vector<TransformationEntry> generatePartialEntries(
    const std::vector<TransformationEntry>& entries)
{
    std::vector<TransformationEntry> result;
    result.reserve(entries.size());
    std::ranges::transform(entries, std::back_inserter(result),
        [](const TransformationEntry& e) { return e.partial(); });
    return result;
}

class OutputPathBuilder {
public:
    // Formats a float with up to 2 decimal places, no trailing zeros
    static std::string formatProportion(const float value) {
        char buf[16];
        int len = std::snprintf(buf, sizeof(buf), "%.3f", value);
        while (len > 0 && buf[len - 1] == '0') --len;
        if (len > 0 && buf[len - 1] == '.') --len;
        return {buf, static_cast<size_t>(len)};
    }

    // Builds a standard path prefix
    static std::string buildStandard(const std::string& outputDir,
                                     const std::string& baseName,
                                     const std::string& suffix,
                                     const float proportion) {
        return std::format("{}{} - {} {} % ", outputDir, baseName, suffix, formatProportion(100.f * proportion));
    }

    // Simple image paths
    static std::string image_edge_detector(const std::string& baseName) {
        return std::format("{}{} - GT.png", folder_edgedetector, baseName);
    }

    static std::string image_one_color(const std::string& baseName, const std::string& colors, int tolerance, size_t idx) {
        return std::format("{}{} - Tolerance {} - {} - {}-{}.png",
                           folder_onecolor,
                           baseName,
                           tolerance,
                           colors,
                           idx,
                           parameters::numProportionSteps - 1);
    }

    // Complete / reverse / partial paths
    static std::string image_complete(const std::string& outputDir,
                                const std::string& baseName,
                                const std::string& suffix,
                                const float proportion,
                                int colorNuance) {
        return buildStandard(outputDir, baseName, suffix, proportion) + std::format("- CN {}.png", colorNuance);
    }

    static std::string image_reverse(const std::string& outputDir,
                               const std::string& baseName,
                               const std::string& suffix,
                               const float proportion) {
        return buildStandard(outputDir, baseName, suffix, proportion) + ".png";
    }

    static std::string image_partial(const std::string& outputDir,
                               const std::string& baseName,
                               const std::string& suffix,
                               const float proportion,
                               int colorNuance,
                               const bool diagonal,
                               const std::vector<int>& rectangles) {
        const std::string rectInfo = diagonal
            ? std::format("- {} Squares", rectangles.size() - 1)
            : std::format("- Rectangles {} - {}", rectangles.front(), rectangles.back());
        return buildStandard(outputDir, baseName, suffix, proportion) + rectInfo + std::format(
                   " - CN {}.png", colorNuance);
    }

    // Video paths
    static std::string video_several_colors(const std::string& baseName, int nFrames) {
        return std::format("{}{} - {} images - {} fps - {{{}-{}-{}}}.mp4",
                           folder_videos, baseName, nFrames, parameters::fps,
                           parameters::colorNuances[0],
                           parameters::colorNuances[1],
                           parameters::colorNuances[2]);
    }

    static std::string video_one_color(const std::string& baseName, size_t nFrames, size_t idx) {
        return std::format("{}{} - {} images - {} fps - {} - {} - {}.mp4",
                                       folder_onecolor, baseName, nFrames,
                                       parameters::fps,
                                       idx,
                                       formatToleranceColors(parameters::toleranceOneColor),
                                       writingWeightedColors(parameters::weightOfRGB, false));
    }

    static std::string video_edge_detector(const std::string& baseName,
                                           int framesToProcess,
                                           const int totalFrames,
                                           const double fps) {
        std::string range = (framesToProcess == totalFrames) ? " - " :
                            std::format("{{{}-{}}} ", parameters::frames[0], parameters::frames[1]);
        return std::format("{}{} - {} frames {}{} fps.mp4",
                           folder_edgedetector, baseName, framesToProcess, range,
                           formatProportion(static_cast<float>(fps)));
    }

    static std::string video_edge_detector_temp(const std::string& baseName) {
        return std::format("{}{}_temp.mp4", folder_edgedetector, baseName);
    }

    static std::string formatToleranceColors(const std::span<const int> toleranceOneColor) {
        std::string s = "{";
        for (size_t i = 0; i < 3; ++i) {
            if (i > 0) s += '-';
            s += std::to_string(toleranceOneColor[i]);

        }
        s += '}';
        return s;
    }
    // Weighted RGB as "{r-g-b}"
    static std::string writingWeightedColors(const std::span<const float> weightOfRGB, const bool integerMode) {
        std::string s = "{";
        for (size_t i = 0; i < 3; ++i) {
            if (i > 0) s += '-';
            s += integerMode ? std::to_string(static_cast<int>(weightOfRGB[i]))
                             : formatProportion(weightOfRGB[i]);
        }
        s += '}';
        return s;
    }
};

class ProgressNotifier {
public:
    ProgressNotifier(std::string title, const std::size_t totalWork)
        : title_(std::move(title)),
          total_(totalWork),
          start_(std::chrono::steady_clock::now())
    {}

    ~ProgressNotifier() {
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_).count();

        const std::string message =
            "100% completed\n"
            "Total time: " + std::to_string(elapsed) + " s";
    }

    void update(const std::size_t current) {
        const double ratio = static_cast<double>(current) / total_;
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_).count();

        for (std::size_t i = 0; i < kMilestones.size(); ++i) {
            if (ratio >= kMilestones[i] / 100.0)
                notifyOnce(milestonesSent_[i], kMilestones[i], elapsed);
        }
    }

    void notifyStep(const std::string& stepName,
                    const std::chrono::steady_clock::time_point stepStart) const {
        const auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - stepStart).count();

        const std::string message =
            stepName + " completed\n"
            "Duration: " + std::to_string(duration) + " s";

        sendNotification(title_, message);
    }

private:
    void notifyOnce(std::atomic<bool>& flag, const int percent, const long elapsed) const {
        if (bool expected = false; flag.compare_exchange_strong(expected, true)) {
            const std::string message =
                std::to_string(percent) + "% completed\n"
                "Elapsed: " + std::to_string(elapsed) + " s";
            sendNotification(title_, message);
        }
    }

    static constexpr std::array<int, 4> kMilestones = {20, 40, 60, 80};

    std::string title_;
    std::size_t total_;
    std::array<std::atomic<bool>, kMilestones.size()> milestonesSent_{};
    std::chrono::steady_clock::time_point start_;
};

class Logger {
public:
    // Variadic template: handles any number/type of arguments
    template<typename... Args>
    static void log(Args&&... args) {
        (std::cout << ... << std::forward<Args>(args)) << std::endl;
    }

    // Overload with '\r' + flush for progress lines (like frame counter)
    template<typename... Args>
    static void logProgress(Args&&... args) {
        (std::cout << ... << std::forward<Args>(args)) << "\r";
        std::cout.flush();
    }

    template<typename... Args>
    static void err(Args&&... args) {
        (std::cerr << ... << std::forward<Args>(args)) << std::endl;
    }
};

#endif //IMAGE_PROCESSING_PROCESSINGCONFIG_H
