#include "ImageProcessor.h"
#include <vector>
#include <functional>

using TransformationFunc = std::function<void(Image&, int)>;

void processImageTransforms(const std::string& inputFileName) {
    // Extract the base name
    size_t dotPos = inputFileName.find_last_of('.');
    std::string baseName = inputFileName.substr(0, dotPos);

    // Load the image
    std::string inputPath = "Input/" + inputFileName;
    Image image(inputPath.c_str());

    // Configure transformations
    std::vector<TransformationFunc> transformations = {
        [](Image& img, int i) { img.btb(i); },
        [](Image& img, int i) { img.btw(i); },
        [](Image& img, int i) { img.wtb(i); },
        [](Image& img, int i) { img.wtw(i); },
        [](Image& img, int i) { img.black_to_white(i); },
        [](Image& img, int i) { img.white_to_black(i); }
    };

    std::vector<std::string> suffixes = {"BTB", "BTW", "WTB", "WTW", "BTOW", "WTOB"};
    std::vector<std::string> outputDirs = {
        "Output/BTB/", "Output/BTW/", "Output/WTB/",
        "Output/WTW/", "Output/BTOW/", "Output/WTOB/"
    };

    // Apply transformations
    for (int threshold = 20; threshold <= 240; threshold += 20) {
        for (size_t i = 0; i < transformations.size(); ++i) {
            Image modified = image;
            transformations[i](modified, threshold);

            std::string outputPath = outputDirs[i] + baseName + " - " + suffixes[i] + " " + std::to_string(threshold) + ".png";
            modified.write(outputPath.c_str());

            if (threshold == 120) {
                std::string specialPath = "Output/120/" + baseName + " - " + suffixes[i] + " 120.png";
                modified.write(specialPath.c_str());
            }
        }
    }
}
