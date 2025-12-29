#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "Image.h"

#include <string>
#include <vector>

void processImageTransforms(
    const std::string& inputPath,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    int fraction,
    const std::vector<int>& rectanglesToModify,
    bool totalT,
    bool partialT
);

#endif
