#ifndef VIDEOPROCESSING_H
#define VIDEOPROCESSING_H

#include <vector>
#include <string>

void processVideoTransforms(
    const std::string& baseName,
    const std::string& inputPath,
    int fps,
    const std::vector<float>& proportions,
    const std::vector<int>& colorNuances,
    const std::vector<int>& frames,
    const std::vector<int>&  tolerance,
    const std::vector<float>& weightOfRGB,
    bool average
    );

#endif // VIDEOPROCESSING_H
