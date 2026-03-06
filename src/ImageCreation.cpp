#include "ImageCreation.h"

void processImageTransforms(
    const std::string& baseName,
    const std::string& inputPath
) {

    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected, no transformation applied.");
        return;
    }

    Logger::log(inputPath);

    const Image image(inputPath.c_str());

    edge_detector_image(image, baseName);
    oneColorTransformations(image, baseName);
    complete_transformations_by_proportion(image, baseName);
    reverse_transformations_by_proportion(image, baseName);


    for (const bool useDiagonal : {false, true}) {
        if (useDiagonal ? !parameters::partialInDiagonal : !parameters::partial)
            continue;

        const std::vector<int> rectangles = useDiagonal
            ? genererRectanglesInDiagonal()
            : decode_rectangles().second;

        partial_transformations_by_proportion(image, baseName, rectangles);
    }

}
