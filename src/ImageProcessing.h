#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "Image.h"

#include <string>
#include <vector>

void processImageTransforms(
    const std::string& inputPath,
    const std::string& baseName,
    const std::vector<int>& thresholdsAndStep,
    int fraction,
    const std::vector<int>& rectanglesToModify,
    const std::vector<int>&  tolerance,
    bool totalStepByStepT,
    bool totalBlackAndWhiteT,
    bool totalReversalT,
    bool partialT,
    bool alternatingBlackAndWhite,
    bool oneColor
);

#endif
