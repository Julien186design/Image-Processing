#include "VideoCreation.h"

#include <string>

void processVideoTransforms(
const std::string& baseName,
const std::string& inputPath
) {
    // Checking extension MP4
    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected → edge_detector_video");
        edge_detector_video(baseName, inputPath);
    }
    else {
        Logger::log("Non-MP4 file detected → colored_transformations");
        several_colors_transformations_streaming(baseName, inputPath);
        one_color_transformations_streaming(baseName, inputPath);
    }
}
