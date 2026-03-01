#pragma once

#ifndef IMAGE_PROCESSING_PROCESSINGCONFIG_H
#define IMAGE_PROCESSING_PROCESSINGCONFIG_H

#define THREAD_OFFSET 1

#include  "Image.h"

#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <omp.h>


struct PropRange {
    float start;
    float stop;
    float step;

    struct iterator {
        float current;
        float step;

        float operator*() const noexcept {
            return current;
        }

        iterator& operator++() noexcept {
            current += step;
            return *this;
        }
    };

    struct sentinel {
        float stop;
    };

    [[nodiscard]]
    iterator begin() const noexcept {
        return { start, step };
    }

    [[nodiscard]]
    sentinel end() const noexcept {
        return { stop };
    }
};

inline bool operator!=(const PropRange::iterator& it,
                       const PropRange::sentinel& s) noexcept {
    return it.current <= s.stop + 1e-6f;
}

struct parameters {
    const PropRange proportions {0.f, 1.f, 0.125f}; // {first proportion, last proportion, step}
    const std::vector<int> colorNuances = {0, 0, 20}; // {first color, last color, step}
    const std::vector<int> frames = {0, 100};
    static constexpr int fps = 60;
    static constexpr int fraction = 2;
    std::vector<int> rectangles = {8, 12};
    const std::vector<int> toleranceOneColor = {1, 2, 1};
    const std::vector<float> weightOfRGB = {0, 1, 0.1};
    static constexpr bool complete_transformation_colors_by_proportion = false;
    static constexpr bool oneColor = true;
    static constexpr bool average = false;
    static constexpr bool totalReversal = true;
    static constexpr bool partial = false;
    static constexpr bool partialInDiagonal = false;
};

constexpr const char* output_folder = "Output/";
const std::string folder_120        = std::string(output_folder) + "120/";
const std::string folder_videos     = std::string(output_folder) + "Videos/";
const std::string folder_edgedetector = std::string(output_folder) + "Edge Detector/";
const std::string folder_onecolor = std::string(output_folder) + "One Color/";

void image_and_video_processing(const std::string& inputFile);

inline void sendNotification(const std::string& title, const std::string &message)
{
    const std::string command =
        "notify-send '" + title + "' '" + message + "'";

    (void)std::system(command.c_str());
}

inline std::vector<int> genererRectanglesInDiagonal(const int fraction) {
    if (fraction <= 0) return {};
    const int taille = 1 << fraction;  // 2^fraction via bit-shift
    const int pas = taille + 1;

    std::vector<int> vector;
    vector.reserve(taille);

    vector.push_back(-1);
    for (int i = 0; i < taille; ++i) {
        vector.push_back(i * pas);
    }
    return vector;
}

inline std::pair<bool, std::vector<int>> decode_rectangles(const std::vector<int>& input) {
    if (input.empty()) return {false, {}};

    if (input[0] == -1) {
        // Diagonal case: rectangles are explicitly listed after the flag
        return {true, std::vector<int>(input.begin() + 1, input.end())};
    }

    // Non-diagonal case: input is {start, end}, expand to range
    if (input.size() != 2) return {false, {}};
    const int start = std::min(input[0], input[1]);
    const int end   = std::max(input[0], input[1]);

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

class OutputPathBuilder {
public:
    // Formats a float with up to 2 decimal places, no trailing zeros
    static std::string formatProportion(float value) {
        std::string s = std::format("{:.4f}", value);
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s.pop_back();
        return s;
    }

    // Writes the standard prefix directly into an existing stream (avoids intermediate string)
    static void writeStandard(
        std::ostringstream& oss,
        const std::string& outputDir,
        const std::string& baseName,
        const std::string& suffix,
        const float proportion
    ) {
        oss << outputDir << baseName << " - " << suffix
            << ' ' << formatProportion(100.f * proportion) << " % ";
    }

    static std::string image_edge_detector(
        const std::string& baseName
    ) {
        std::ostringstream oss;
        oss << folder_edgedetector << baseName << "GT.png";
        return oss.str();
    }

    static std::string image_one_color(
        const bool average,
        const std::string& baseName,
        const std::string& colors,
        const int tolerance
    ) {
        std::ostringstream oss;
        oss << folder_onecolor << baseName << " - Average ";
        if (average)
            oss << "1";
        else
            oss << "0";
        oss << " - Tolerance " << std::to_string(tolerance) << colors << ".png";
        return oss.str();
    }

    static std::string complete(
        const std::string& outputDir,
        const std::string& baseName,
        const std::string& suffix,
        const float proportion,
        const int colorNuance
    ) {
        std::ostringstream oss;
        writeStandard(oss, outputDir, baseName, suffix, proportion);
        oss << "- CN " << colorNuance << ".png";
        return oss.str();
    }

    static std::string reverse(
        const std::string& outputDir,
        const std::string& baseName,
        const std::string& suffix,
        const float proportion
    ) {
        std::ostringstream oss;
        writeStandard(oss, outputDir, baseName, suffix, proportion);
        oss << ".png";
        return oss.str();
    }

    static std::string partial(
        const std::string& outputDir,
        const std::string& baseName,
        const std::string& suffix,
        const float proportion,
        const int colorNuance,
        const bool diagonal,
        const std::vector<int>& rectangles
    ) {
        std::ostringstream oss;
        writeStandard(oss, outputDir, baseName, suffix, proportion);
        if (diagonal)
            oss << "- " << rectangles.size() << " Squares";
        else
            oss << "- Rectangles " << rectangles.front() << " - " << rectangles.back();
        oss << " - CN " << colorNuance << ".png";
        return oss.str();
    }

    static std::string build120(
        const std::string& folder120,
        const std::string& baseName,
        const std::string& transformationType,
        const std::string& suffix
    ) {
        // Simple concatenation — no stream needed
        return folder120 + baseName + transformationType + suffix;
    }

    static std::string video_several_colors(
        const std::string& baseName,
        const int nFrames,
        const int fps,
        const std::vector<int>& colorNuances
    ) {
        std::ostringstream oss;
        oss << folder_videos << baseName
            << " - " << std::to_string(nFrames) << " images"
            << " - " << fps     << " fps"
            << " - {" << std::to_string(colorNuances[0]) << '-' <<
                std::to_string(colorNuances[1]) << '-' << std::to_string(colorNuances[2]) << '}'
            << ".mp4";
        return oss.str();
    }

    static std::string video_one_color(
        const std::string& baseName,
        const size_t nFrames,
        const int fps,
        const std::vector<float>& weightOfRGB,
        const bool average
    ) {
        std::ostringstream oss;
        oss << folder_videos << baseName
            << " - One Color - "
            << nFrames << " images - "
            << fps     << " fps -"
            << writingWeightedColors(weightOfRGB, false);
        if (average) oss << " Average";
        oss << ".mp4";
        return oss.str();
    }

    static std::string video_edge_detector(
        const std::string& baseName,
        const int framesToProcess,
        const int totalFrames,
        const double fps,
        const std::vector<int>& frames
    ) {
        std::ostringstream oss;
        oss << folder_edgedetector << baseName
            << " - Edge Detector - "
            << std::to_string(framesToProcess) << " frames ";
        if (framesToProcess == totalFrames)
            oss << " - ";
        else
            oss << '{' << frames[0] << '-' << frames[1] << "} ";
        oss << formatProportion(static_cast<float>(fps)) << " fps.mp4";
        return oss.str();
    }

    // Formats weighted RGB values as " {r-g-b}"
    static std::string writingWeightedColors(const std::vector<float>& weightOfRGB, const bool integerMode) {
        std::string s;
        s.reserve(32);
        s += " {";
        for (size_t i = 0; i < 3; ++i) {
            if (i > 0) s += '-';
            const float val = weightOfRGB[i];
            if (integerMode)
                s += std::to_string(static_cast<int>(val));
            else
                s += formatProportion(val);
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

#endif //IMAGE_PROCESSING_PROCESSINGCONFIG_H
