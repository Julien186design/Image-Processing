#include "ImageProcessing.h"
#include "TransformationsConfig.h"
#include <vector>
#include <functional>
#include <string>
#include <cstring>



void applyAndSaveTransformations(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    bool saveAt120 = true,
    const std::string& folder120 = DEFAULT_120_FOLDER
    )
{
    Image modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int t = threshold; t <= lastThreshold; t += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {

            memcpy(modified.data, baseImage.data, baseImage.size);

            transforms[i](modified, t);

            std::string outputPath = outputDirs[i] + baseName + " - "
                + suffixes[i] + " " + std::to_string(t) + ".png";
            modified.write(outputPath.c_str());

            if (saveAt120 && t == 120) {
                std::string specialPath = folder120 + baseName
                    + " - " + suffixes[i] + " 120.png";
                modified.write(specialPath.c_str());
            }
        }
    }
}

void applyAndSaveReversalTransformations(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    bool saveAt120 = true,
    const std::string& folder120 = DEFAULT_120_FOLDER
    )
{
    Image modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int t = threshold; t <= lastThreshold; t += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {

            memcpy(modified.data, baseImage.data, baseImage.size);

            transforms[i](modified, t);

            std::string outputPath = outputDirs[i] + baseName + " Reversal - "
                + suffixes[i] + " " + std::to_string(t) + ".png";
            modified.write(outputPath.c_str());

            if (saveAt120 && t == 120) {
                std::string specialPath = "Output/120/" + baseName
                    + " Reversal - " + suffixes[i] + " 120.png";
                modified.write(specialPath.c_str());
            }
        }
    }
}


void applyAndSaveAlternatingTransformations(
    const Image& baseImage,
    const std::vector<AlternatingTransformation>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    bool saveAt120 = true,
    const std::string& folder120 = DEFAULT_120_FOLDER
    )
{
    Image modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int t = threshold; t <= lastThreshold; t += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {

            memcpy(modified.data, baseImage.data, baseImage.size);

            transforms[i](modified, t, t, t);

            std::string outputPath = outputDirs[i] + baseName + " Alternating - "
                + suffixes[i] + " " + std::to_string(t) + ".png";
            modified.write(outputPath.c_str());

            if (saveAt120 && t == 120) {
                std::string specialPath = folder120 + baseName
                    + " Alternating - " + suffixes[i] + " 120.png";
                modified.write(specialPath.c_str());
            }
        }

    }
}


void applyAndSavePartialTransformations(
    const Image& baseImage,
    const std::vector<PartialTransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    int fraction,
    const std::vector<int>& rectanglesToModify)
{
    Image modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int t = threshold; t <= lastThreshold; t += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {

            memcpy(modified.data, baseImage.data, baseImage.size);

            transforms[i](modified, t, fraction, rectanglesToModify);

            std::string outputPath = outputDirs[i] + baseName + " - "
                + suffixes[i] + " " + std::to_string(t) + ".png";
            modified.write(outputPath.c_str());
        }
    }
}

void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& tolerance)
{
    const int start = tolerance[0];
    const int end   = tolerance[1];
    const int step  = tolerance[2];

    for (int t = start; t <= end; t += step) {
        Image oneColorImage = baseImage;
        oneColorImage.simplify_to_dominant_color_combinations(t);
        std::string oneColorOutputPath;
        oneColorOutputPath = "Output/One Color/" + baseName + " - Tolerance " + std::to_string(t) +
                             ".png";
        oneColorImage.write(oneColorOutputPath.c_str());
    }
}

void processImageTransforms(
    const std::string& inputPath,
    const std::string& baseName,
    int first_threshold,
    int last_threshold,
    int step,
    int fraction,
    const std::vector<int>& rectanglesToModify,
    const std::vector<int>& tolerance,
    bool totalStepByStepT,
    bool totalBlackAndWhiteT,
    bool totalReversalT,
    bool partialT,
    bool alternatingBlackAndWhite,
    bool oneColor
    )
{
    Image image(inputPath.c_str());

    // One Color transformation
    if (oneColor) {
        oneColorTransformations(image, baseName, tolerance);
    }



    if (totalStepByStepT) {
        applyAndSaveTransformations(
            image,
            total_step_by_step_transformations,
            total_step_by_stepoutput_dirs,
            total_step_by_step_suffixes,
            baseName,
            first_threshold,
            last_threshold,
            step,
            true,  // saveAt120
            DEFAULT_120_FOLDER
        );
    }

    if (totalReversalT) {
        applyAndSaveReversalTransformations(
            image,
            total_reversal_step_by_step_transformations,
            total_step_by_stepoutput_dirs,
            total_step_by_step_suffixes,
            baseName,
            first_threshold,
            last_threshold,
            step,
            true,  // saveAt120
            DEFAULT_120_FOLDER
        );
    }

    if (alternatingBlackAndWhite) {
        applyAndSaveAlternatingTransformations(
            image,
            total_alternating_black_and_white_transformations,
            total_step_by_stepoutput_dirs,
            total_step_by_step_suffixes,
            baseName,
            first_threshold,
            last_threshold,
            step,
            true,  // saveAt120
            DEFAULT_120_FOLDER
        );
    }

    if (totalBlackAndWhiteT) {
        applyAndSaveTransformations(
            image,
            total_black_and_white_transformations,
            total_black_and_white_output_dirs,
            total_black_and_white_suffixes,
            baseName,
            first_threshold,
            last_threshold,
            step,
            true,  // saveAt120
            DEFAULT_120_FOLDER
        );
    }

    if (partialT) {
        applyAndSavePartialTransformations(
            image,
            partialTransformationsFunc,
            generatePartialOutputDirs(),
            generatePartialSuffixes(),
            baseName,
            first_threshold,
            last_threshold,
            step,
            fraction,
            rectanglesToModify
        );
    }
}
