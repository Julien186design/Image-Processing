#include "ImageProcessing.h"
#include <vector>
#include <functional>
#include <string>
#include <algorithm>


using TransformationFunc = std::function<void(Image&, int)>;
using PartialTransformationFunc = std::function<void(Image&, int, int, const std::vector<int>&)>;

void processImageTransforms(const std::string& inputPath, const std::string& baseName,
    int threshold, int lastThreshold, int step, int fraction, const std::vector<int>& rectanglesToModify,
    bool totalT, bool partialT) {

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
    std::vector<std::string> suffixes = {"BTB", "BTW", "WTB", "WTW", "BTOW", "WTOB"};
    std::vector<std::string> partialSuffixes;

    std::transform(
        suffixes.begin(), suffixes.end(),
        std::back_inserter(partialSuffixes),
        [](const std::string& s) { return s + " Partial"; }
    );


    // Define output directories for each transformation
    std::vector<std::string> outputDirs = {
        "Output/BTB/", "Output/BTW/", "Output/WTB/",
        "Output/WTW/", "Output/BTOW/", "Output/WTOB/"
    };

    std::vector<std::string> partialOutputDirs;

    std::transform(
        outputDirs.begin(), outputDirs.end(),
        std::back_inserter(partialOutputDirs),
        [](const std::string& s) {
            // Vérifie si la chaîne n'est pas vide pour éviter les erreurs
            if (s.empty()) {
                return s;
            }
            // Crée une nouvelle chaîne sans le dernier caractère
            return s.substr(0, s.size() - 1) + " Square/";
        }
    );
    

    if (totalT) {
        for (int t = threshold; t <= lastThreshold; t += step) {
            for (size_t i = 0; i < transformations.size(); ++i) {
                Image modified = image;
                transformations[i](modified, t); // 0.0f car pas utilisé

                std::string outputPath = outputDirs[i] + baseName + " - " + suffixes[i]
                    + " " + std::to_string(t) + ".png";
                modified.write(outputPath.c_str());

                if (t == 120) {
                    std::string specialPath = "Output/120/" + baseName + " - " + suffixes[i] + " 120.png";
                    modified.write(specialPath.c_str());
                }
            }
        }
    }

    if (partialT) {
        for (int t = threshold; t <= lastThreshold; t += step) {
            for (size_t i = 0; i < partialTransformationsFunc.size(); ++i) {
                Image modified = image;
                partialTransformationsFunc[i](modified, t, fraction, rectanglesToModify);

                std::string outputPath = partialOutputDirs[i] + baseName + " - " + partialSuffixes[i]
                    + " " + std::to_string(t) + ".png";
                modified.write(outputPath.c_str());

            }
        }
    }

}
