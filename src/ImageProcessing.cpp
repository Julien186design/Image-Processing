#include "ImageProcessing.h"
#include <vector>
#include <functional>
#include <string>
#include <algorithm>
#include <cstring>

using TransformationFunc = std::function<void(Image&, int)>;
using PartialTransformationFunc = std::function<void(Image&, int, int, const std::vector<int>&)>;

void applyAndSaveTransformations(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    bool saveAt120 = true)
{
    Image modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int t = threshold; t <= lastThreshold; t += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {
            // Restaurer l'image originale (copie optimisée)
            memcpy(modified.data, baseImage.data, baseImage.size);

            // Appliquer la transformation
            transforms[i](modified, t);

            // Sauvegarder
            std::string outputPath = outputDirs[i] + baseName + " - "
                + suffixes[i] + " " + std::to_string(t) + ".png";
            modified.write(outputPath.c_str());

            // Sauvegarde spéciale pour t=120
            if (saveAt120 && t == 120) {
                std::string specialPath = "Output/120/" + baseName
                    + " - " + suffixes[i] + " 120.png";
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
            // Restaurer l'image originale
            memcpy(modified.data, baseImage.data, baseImage.size);

            // Appliquer la transformation partielle
            transforms[i](modified, t, fraction, rectanglesToModify);

            // Sauvegarder
            std::string outputPath = outputDirs[i] + baseName + " - "
                + suffixes[i] + " " + std::to_string(t) + ".png";
            modified.write(outputPath.c_str());
        }
    }
}

void processImageTransforms(
    const std::string& inputPath,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    int fraction,
    const std::vector<int>& rectanglesToModify,
    bool totalT,
    bool partialT)
{
    Image image(inputPath.c_str());

    // Define the transformation functions
    std::vector<TransformationFunc> transformations = {
        [](Image& img, int i) { img.darkenBelowThreshold(i); },
        [](Image& img, int i) { img.whitenBelowThreshold(i); },
        [](Image& img, int i) { img.darkenAboveThreshold(i); },
        [](Image& img, int i) { img.whitenAboveThreshold(i); },
        [](Image& img, int i) { img.black_to_white(i); },
        [](Image& img, int i) { img.white_to_black(i); }
    };

    std::vector<PartialTransformationFunc> partialTransformationsFunc = {
        [](Image& img, int i, int fraction, const std::vector<int>& rectanglesToModify) {
            img.darkenBelowThresholdRegionFraction(i, fraction, rectanglesToModify);
        },
        [](Image& img, int i, int fraction, const std::vector<int>& rectanglesToModify) {
            img.whitenBelowThresholdRegionFraction(i, fraction, rectanglesToModify);
        },
        [](Image& img, int i, int fraction, const std::vector<int>& rectanglesToModify) {
            img.darkenAboveThresholdRegionFraction(i, fraction, rectanglesToModify);
        },
        [](Image& img, int i, int fraction, const std::vector<int>& rectanglesToModify) {
            img.whitenAboveThresholdRegionFraction(i, fraction, rectanglesToModify);
        }
    };

    // Define suffixes for the output file names
    std::vector<std::string> suffixes = {
        "BTB", "BTW", "WTB", "WTW",
        "Reversed black and white", "Original black and white"
    };

    std::vector<std::string> partialSuffixes;
    std::transform(
        suffixes.begin(), suffixes.end(),
        std::back_inserter(partialSuffixes),
        [](const std::string& s) { return s + " Partial"; }
    );

    // Define output directories for each transformation
    std::vector<std::string> outputDirs = {
        "Output/BTB/", "Output/BTW/", "Output/WTB/",
        "Output/WTW/", "Output/Reversed black and white/", "Output/Original black and white/"
    };

    std::vector<std::string> partialOutputDirs;
    std::transform(
        outputDirs.begin(), outputDirs.end(),
        std::back_inserter(partialOutputDirs),
        [](const std::string& s) {
            if (s.empty()) return s;
            return s.substr(0, s.size() - 1) + " Square/";
        }
    );

    // One Color transformation
    Image oneColorImage = image;
    oneColorImage.one_color_at_a_time_and_thoroughly();
    std::string oneColorOutputPath = "Output/One Color/" + baseName + " - One Color.png";
    oneColorImage.write(oneColorOutputPath.c_str());


    if (totalT) {
        applyAndSaveTransformations(
            image,
            transformations,
            outputDirs,
            suffixes,
            baseName,
            threshold,
            lastThreshold,
            step,
            true  // saveAt120
        );
    }

    if (partialT) {
        applyAndSavePartialTransformations(
            image,
            partialTransformationsFunc,
            partialOutputDirs,
            partialSuffixes,
            baseName,
            threshold,
            lastThreshold,
            step,
            fraction,
            rectanglesToModify
        );
    }
}
