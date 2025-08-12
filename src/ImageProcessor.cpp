#include "ImageProcessoring.h"
#include <vector>
#include <functional>

#include <vector>
#include <string>


using TransformationFunc = std::function<void(Image&, int)>;

void processImageTransforms(const std::string& inputPath, const std::string& baseName) {

    Image image(inputPath.c_str());

    // Define the transformation functions
    std::vector<TransformationFunc> transformations = {
        [](Image& img, int i) { img.btb(i); },
        [](Image& img, int i) { img.btw(i); },
        [](Image& img, int i) { img.wtb(i); },
        [](Image& img, int i) { img.wtw(i); },
        [](Image& img, int i) { img.black_to_white(i); },
        [](Image& img, int i) { img.white_to_black(i); }
    };

    // Define suffixes for the output file names
    std::vector<std::string> suffixes = {"BTB", "BTW", "WTB", "WTW", "BTOW", "WTOB"};

    // Define output directories for each transformation
    std::vector<std::string> outputDirs = {
        "Output/BTB/", "Output/BTW/", "Output/WTB/",
        "Output/WTW/", "Output/BTOW/", "Output/WTOB/"
    };

    // Apply each transformation for a range of threshold values
    for (int threshold = 20; threshold <= 240; threshold += 20) {
        for (size_t i = 0; i < transformations.size(); ++i) {
            Image modified = image; // Create a copy of the original image
            transformations[i](modified, threshold); // Apply the transformation

            // Define the output path for the modified image
            std::string outputPath = outputDirs[i] + baseName + " - " + suffixes[i] + " " + std::to_string(threshold) + ".png";
            modified.write(outputPath.c_str()); // Save the modified image

            // Special case for threshold 120
            if (threshold == 120) {
                std::string specialPath = "Output/120/" + baseName + " - " + suffixes[i] + " 120.png";
                modified.write(specialPath.c_str()); // Save the modified image to the special directory
            }
        }
    }

}
