#include "ProcessingConfig.h"
#include "ImageProcessing.h"
#include "VideoProcessing.h"

#include <chrono>

void image_and_video_processing(const std::string& inputFile) {

    const auto start = std::chrono::high_resolution_clock::now();

    const auto [baseName, inputPath] = extractImageInfo(inputFile);

    const parameters cfg;

    processImageTransforms(baseName, inputPath, cfg.proportions, cfg.colorNuances,
        cfg.rectangles, cfg.toleranceOneColor, cfg.weightOfRGB);

    processVideoTransforms(baseName, inputPath, cfg.proportions, cfg.colorNuances,
                           cfg.frames, cfg.toleranceOneColor, cfg.weightOfRGB);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    const std::string message = baseName + "\nThe image and video processing has finished in " +
        std::to_string(duration.count()) + " seconds.";
    sendNotification("Program Execution Complete", message);


}
