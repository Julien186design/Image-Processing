#include "ImageProcessoring.h"

int main() {
    const std::string inputFile = "Oreo cake.jpg";	//image to process
    const std::string folderName = "Cake";	//folder
    
    // Extract the base name without the extension
    size_t dotPos = inputFile.find_last_of('.');
    std::string baseName = inputFile.substr(0, dotPos);
    
    // Load the image from the input path
    std::string inputPath = "Input/" + folderName + "/" + inputFile;
    printf("%s\n", inputPath.c_str()); // Print the input path
    printf("%s\n", folderName.c_str()); // Print the folder name
    
    
	processImageTransforms(inputPath, baseName);
	
    return 0;
}
