#include "ProcessingConfig.h"
#include "ImageCreation.h"
#include "VideoCreation.h"

#include <chrono>

// Définitions des variables déclarées extern dans le header
const std::string output_folder          = "Output/";
const std::string folder_50          = std::string(output_folder) + "50/";
const std::string folder_videos       = std::string(output_folder) + "Videos/";
const std::string folder_edgedetector = std::string(output_folder) + "Edge Detector/";
const std::string folder_onecolor     = std::string(output_folder) + "One Color/";

const std::vector<TransformationEntry> total_step_by_step_entries = {
    { "BTB", std::string(output_folder) + "BTB/" },
    { "BTW", std::string(output_folder) + "BTW/" },
    { "WTB", std::string(output_folder) + "WTB/" },
    { "WTW", std::string(output_folder) + "WTW/" },
};

const std::vector<TransformationEntry> reversal_step_by_step_entries = {
    { "Reversal-BT", std::string(output_folder) + "Reversal-BT/" },
    { "Reversal-WT", std::string(output_folder) + "Reversal-WT/" },
};

const std::vector<TransformationEntry> total_black_and_white_entries = {
    { "Black and white - Original", std::string(output_folder) + "Original black and white/" },
    { "Black and white - Reversed", std::string(output_folder) + "Reversed black and white/" },
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
