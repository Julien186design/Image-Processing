#ifndef VIDEOPROCESSING_H
#define VIDEOPROCESSING_H

#include <vector>
#include <string>


void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath,
    int fps,
    const std::vector<float> &proportions,
    const std::vector<int>& colorNuances
);

void edge_detector_video(
    const std::string& baseName,
    const std::string& inputVideoPath,
    const std::vector<int>& frames

);

void processVideoTransforms(
    const std::string& baseName,
    const std::string& inputPath,
    int fps,
    const std::vector<float>& proportions,
    const std::vector<int>& colorNuances,
    const std::vector<int>& frames
    );

#endif // VIDEOPROCESSING_H
