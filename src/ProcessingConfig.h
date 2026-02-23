#ifndef IMAGE_PROCESSING_PROCESSINGCONFIG_H
#define IMAGE_PROCESSING_PROCESSINGCONFIG_H

#include "ImageProcessing.h"
#include <vector>

struct parameters {
    const std::vector<float> proportions = {0, 1, 0.125}; // {first proportion, last proportion, step}
    const std::vector<int> colorNuances = {0, 40, 40}; // {first color, last color, step}
    const std::vector<int> frames = {0, 0};
    const int fps = 30;
    const int fraction = 1;
    std::vector<int> rectanglesToModify = range_to_vector({1, 3});
    const std::vector<int> toleranceOneColor = {0, 0, 1};
    const std::vector<float> weightOfRGB = {0, 1, 0.2};
    const bool severalColorsByProportion = false;
    const bool oneColor = true;
    const bool average = false;
    const bool totalReversal = false;
    const bool partial = false;
    const bool partialInDiagonal = false;
};

void image_and_video_processing(
    const std::string& baseName,
    const std::string& inputPath);

#endif //IMAGE_PROCESSING_PROCESSINGCONFIG_H