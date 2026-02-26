#include "ProcessingConfig.h"

#include "ImageProcessing.h"
#include "VideoProcessing.h"

void image_and_video_processing(
    const std::string& baseName,
    const std::string& inputPath) {

    const parameters cfg;

    processImageTransforms(baseName, inputPath, cfg.proportions, cfg.colorNuances, cfg.fraction,
        cfg.rectangles, cfg.toleranceOneColor, cfg.weightOfRGB, cfg.severalColorsByProportion,
        cfg.totalReversal, cfg.partial, cfg.partialInDiagonal, cfg.oneColor, cfg.average);

    processVideoTransforms(baseName, inputPath, cfg.fps, cfg.proportions, cfg.colorNuances,
                           cfg.frames, cfg.toleranceOneColor, cfg.weightOfRGB, cfg.average);



}
