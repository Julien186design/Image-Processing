#include "ProcessingConfig.h"
#include "ImageCreation.h"
#include "VideoCreation.h"

#include <chrono>

// Définitions des variables déclarées extern dans le header
const std::string folder_output          = "Output/";
const std::string folder_50          = std::string(folder_output) + "50/";
const std::string folder_videos       = std::string(folder_output) + "Videos/";
const std::string folder_edgedetector = std::string(folder_output) + "Edge Detector/";
const std::string folder_onecolor     = std::string(folder_output) + "One Color/";

const std::vector<TransformationEntry> total_step_by_step_entries = {
    { "BTB", std::string(folder_output) + "BTB/" },
    { "BTW", std::string(folder_output) + "BTW/" },
    { "WTB", std::string(folder_output) + "WTB/" },
    { "WTW", std::string(folder_output) + "WTW/" },
};

const std::vector<TransformationEntry> reversal_step_by_step_entries = {
    { "Reversal-BT", std::string(folder_output) + "Reversal-BT/" },
    { "Reversal-WT", std::string(folder_output) + "Reversal-WT/" },
};

const std::vector<TransformationEntry> total_black_and_white_entries = {
    { "Black and white - Original", std::string(folder_output) + "Original black and white/" },
    { "Black and white - Reversed", std::string(folder_output) + "Reversed black and white/" },
};

void image_and_video_processing(const std::string& inputFile)
{
    const auto [baseName, inputPath] = extractImageInfo(inputFile);

    const ProgressNotifier notifier("Process Transform", 2);

    auto stepStart = std::chrono::steady_clock::now();

    if (processImageTransforms(baseName, inputPath)) {
        notifier.notifyStep("processImageTransforms", stepStart);
    }

    stepStart = std::chrono::steady_clock::now();
    processVideoTransforms(baseName, inputPath);
    notifier.notifyStep("processVideoTransforms", stepStart);
}
