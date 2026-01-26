#ifndef VIDEOPROCESSING_H
#define VIDEOPROCESSING_H

#include <vector>
#include <string>

void createVideoFromImageList(const std::vector<std::string>& imagePaths, const std::string& outputVideoPath, int fps = 24);

void several_colors_transformations_in_video(const std::vector<std::string>& imageFiles, const std::string& outputVideoPath, int fps = 24);

void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath,
    int fps,
    const std::vector<float> &proportions,
    const std::vector<int>& colorNuances
);

#endif // VIDEOPROCESSING_H
