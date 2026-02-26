#pragma once

#ifndef IMAGE_PROCESSING_PROCESSINGCONFIG_H
#define IMAGE_PROCESSING_PROCESSINGCONFIG_H

#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <chrono>

struct parameters {
    const std::vector<float> proportions = {0, 1, 0.125}; // {first proportion, last proportion, step}
    const std::vector<int> colorNuances = {0, 40, 40}; // {first color, last color, step}
    const std::vector<int> frames = {0, 0};
    const int fps = 60;
    const int fraction = 2;
    std::vector<int> rectangles = {4, 12};
    const std::vector<int> toleranceOneColor = {0, 0, 1};
    const std::vector<float> weightOfRGB = {0, 1, 0.1};
    const bool severalColorsByProportion = false;
    const bool oneColor = true;
    const bool average = false;
    const bool totalReversal = false;
    const bool partial = true;
    const bool partialInDiagonal = false;
};

void image_and_video_processing(
    const std::string& baseName,
    const std::string& inputPath);

/*
 * Sends a desktop notification using notify-send.
 */
inline void sendNotification(
    const std::string& title,
    const std::string& message)
{
    const std::string command =
        "notify-send '" + title + "' '" + message + "'";
    (void)std::system(command.c_str());
}

/*
 * Tracks progress and sends notifications
 * at 25%, 50% and 75%.
 *
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