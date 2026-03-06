#pragma once

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

struct parameters {
    static constexpr std::array<float, 3> proportions = {0.f, 1.f, 1.f};
    static constexpr std::array<int, 3> colorNuances = {0, 60, 5}; // {first color, last color, step}
    static constexpr std::array<int, 2> frames = {0, 0};
    static constexpr int fps = 30;
    static constexpr int fraction = 2;
    static constexpr std::array<int, 2> rectangles = {4, 12};
    static constexpr std::array<int, 3> toleranceOneColor = {0, 40, 1};
    static constexpr std::array<float, 3> weightOfRGB = {0.f, 1.f, .25f};
    static constexpr bool complete_transformation_colors_by_proportion = false;
    static constexpr bool oneColor = true;
    static constexpr bool average = false;
    static constexpr bool totalReversal = false;
    static constexpr bool partial = false;
    static constexpr bool partialInDiagonal = false;
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



constexpr const char* output_folder = "Output/";
const std::string folder_120        = std::string(output_folder) + "120/";
const std::string folder_videos     = std::string(output_folder) + "Videos/";
const std::string folder_edgedetector = std::string(output_folder) + "Edge Detector/";
const std::string folder_onecolor = std::string(output_folder) + "One Color/";

// ─── Step-by-step transformations (BTB, BTW, WTB, WTW) ───────────────────────

const std::vector<TransformationEntry> total_step_by_step_entries = {
    { "BTB", std::string(output_folder) + "BTB/" },
    { "BTW", std::string(output_folder) + "BTW/" },
    { "WTB", std::string(output_folder) + "WTB/" },
    { "WTW", std::string(output_folder) + "WTW/" },
};

// ─── Reversal transformations ─────────────────────────────────────────────────

const std::vector<TransformationEntry> reversal_step_by_step_entries = {
    { "Reversal-BT", std::string(output_folder) + "Reversal-BT/" },
    { "Reversal-WT", std::string(output_folder) + "Reversal-WT/" },
};

// ─── Black-and-white transformations ─────────────────────────────────────────

const std::vector<TransformationEntry> total_black_and_white_entries = {
    { "Black and white - Original", std::string(output_folder) + "Original black and white/" },
    { "Black and white - Reversed", std::string(output_folder) + "Reversed black and white/" },
};

void image_and_video_processing(const std::string& inputFile);

inline void sendNotification(const std::string& title, const std::string &message)
{
    const std::string command =
        "notify-send '" + title + "' '" + message + "'";

    (void)std::system(command.c_str());
}

inline std::vector<int> genererRectanglesInDiagonal() {
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

inline bool is_mp4_file(const std::string& path)
{
    if (path.length() < 4)
        return false;

    std::string ext = path.substr(path.length() - 4);

    // Convert extension to lowercase for case-insensitive comparison
    std::ranges::transform(ext, ext.begin(),
                           [](const unsigned char c) { return std::tolower(c); });

    return ext == ".mp4";
}

inline int computeNumThreads() {
    return std::max(1, omp_get_max_threads() - THREAD_OFFSET);
}


// ─── Helpers to extract suffixes / dirs from an entry list ───────────────────

inline std::vector<std::string> getSuffixes(const std::vector<TransformationEntry>& entries) {
    std::vector<std::string> result;
    result.reserve(entries.size());
    std::ranges::transform(entries, std::back_inserter(result),
        [](const TransformationEntry& e) { return e.suffix; });
    return result;
}

inline std::vector<std::string> getOutputDirs(const std::vector<TransformationEntry>& entries) {
    std::vector<std::string> result;
    result.reserve(entries.size());
    std::ranges::transform(entries, std::back_inserter(result),
        [](const TransformationEntry& e) { return e.output_dir; });
    return result;
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
        int len = std::snprintf(buf, sizeof(buf), "%.2f", value);
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

    static std::string image_one_color(const std::string& baseName, const std::string& colors, int tolerance) {
        return std::format("{}{} - Average {} - Tolerance {}{}.png",
                           folder_onecolor,
                           baseName,
                           parameters::average ? 1 : 0,
                           tolerance,
                           colors);
    }

    // Complete / reverse / partial paths
    static std::string complete(const std::string& outputDir,
                                const std::string& baseName,
                                const std::string& suffix,
                                const float proportion,
                                int colorNuance) {
        return buildStandard(outputDir, baseName, suffix, proportion) + std::format("- CN {}.png", colorNuance);
    }

    static std::string reverse(const std::string& outputDir,
                               const std::string& baseName,
                               const std::string& suffix,
                               const float proportion) {
        return buildStandard(outputDir, baseName, suffix, proportion) + ".png";
    }

    static std::string partial(const std::string& outputDir,
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

    // Simple concatenation
    static std::string build120(const std::string& folder120,
                                const std::string& baseName,
                                const std::string& transformationType,
                                const std::string& suffix) {
        return folder120 + baseName + transformationType + suffix;
    }

    // Video paths
    static std::string video_several_colors(const std::string& baseName, int nFrames) {
        return std::format("{}{} - {} images - {} fps - {{{}-{}-{}}}.mp4",
                           folder_videos, baseName, nFrames, parameters::fps,
                           parameters::colorNuances[0],
                           parameters::colorNuances[1],
                           parameters::colorNuances[2]);
    }

    static std::string video_one_color(const std::string& baseName, size_t nFrames) {
        std::string path = std::format("{}{} - {} images - {} fps - {}",
                                       folder_onecolor, baseName, nFrames, parameters::fps,
                                       writingWeightedColors(parameters::weightOfRGB, false));
        if (parameters::average) path += " Average";
        path += ".mp4";
        return path;
    }

    static std::string video_edge_detector(const std::string& baseName,
                                           int framesToProcess,
                                           const int totalFrames,
                                           const double fps) {
        std::string range = (framesToProcess == totalFrames) ? " - " :
                            std::format("{{{}-{}}} ", parameters::frames[0], parameters::frames[1]);
        return std::format("{}{} - {} frames {}{} fps.mp4",
                           folder_edgedetector, baseName, framesToProcess, range, formatProportion(static_cast<float>(fps)));
    }

    static std::string video_edge_detector_temp(const std::string& baseName) {
        return std::format("{}{}_temp.mp4", folder_edgedetector, baseName);
    }

    // Weighted RGB as "{r-g-b}"
    static std::string writingWeightedColors(const std::span<const float> weightOfRGB, const bool integerMode) {
        std::string s = " {";
        for (size_t i = 0; i < 3; ++i) {
            if (i > 0) s += '-';
            s += integerMode ? std::to_string(static_cast<int>(weightOfRGB[i]))
                             : formatProportion(weightOfRGB[i]);
        }
        s += '}';
        return s;
    }
};
/*
 * Tracks progress and sends notifications
 * at 25%, 50% and 75%.
 * Thread-safe.
 */
class ProgressNotifier {
public:
    ProgressNotifier(
        std::string  title,
        const std::size_t totalWork)
        : title_(std::move(title)),
          total_(totalWork),
          quarterSent_(false),
          halfSent_(false),
          threeQuarterSent_(false),
          start_(std::chrono::steady_clock::now())
    {}

    void update(const std::size_t current)
    {
        const double ratio =
            static_cast<double>(current) / total_;

        const auto now =
            std::chrono::steady_clock::now();

        const auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(
                now - start_).count();

        if (ratio >= 0.25)
            notifyOnce(quarterSent_, 25, elapsed, ratio);

        if (ratio >= 0.50)
            notifyOnce(halfSent_, 50, elapsed, ratio);

        if (ratio >= 0.75)
            notifyOnce(threeQuarterSent_, 75, elapsed, ratio);
    }

private:
    void notifyOnce(std::atomic<bool>& flag,
                    const int percent,
                    const long elapsed,
                    const double ratio) const {
        if (bool expected = false; flag.compare_exchange_strong(expected, true))
        {
            const std::string message =
                    std::to_string(percent) + "% completed\n"
                    "Elapsed: " + std::to_string(elapsed) + " s";

            sendNotification(title_, message);
        }
    }

    std::string title_;
    std::size_t total_;

    std::atomic<bool> quarterSent_;
    std::atomic<bool> halfSent_;
    std::atomic<bool> threeQuarterSent_;

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
