#include "ImageProcessor.h"

int main() {
    const std::string inputFile = "Barbie holds itself.jpg";	//image to process
    const std::string folderName = "Barbie";	//folder
    processImageTransforms(inputFile, folderName);
    return 0;
}
