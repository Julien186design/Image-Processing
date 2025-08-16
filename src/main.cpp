#include "ImageProcessoring.h"

int main() {
    const std::string inputFile = "Oreo cake.jpg";
    const std::string folderName = "Cake";
    
    int threshold = 40;
    int lastThreshold = 220;
    int step = 20;
    		
    // Extract the base name without the extension
    size_t dotPos = inputFile.find_last_of('.');
    std::string baseName = inputFile.substr(0, dotPos);
    
    // Load the image from the input path
    std::string inputPath = "Input/" + folderName + "/" + inputFile;
    printf("%s\n", inputPath.c_str()); // Print the input path
    printf("%s\n", folderName.c_str()); // Print the folder name
        
	processImageTransforms(inputPath, baseName, threshold, lastThreshold, step);

    return 0;
}
