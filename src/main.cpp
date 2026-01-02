#include "ImageProcessing.h"

#include <chrono>

int main() {
	auto start = std::chrono::high_resolution_clock::now();
    const std::string inputFile = "Oreo Cake.jpg";
    const std::string folderName = "Food";

	int threshold = 10;
	int lastThreshold = 240;
	int step = 10;
	int fraction = 2;
	std::vector<int> rectanglesToModify = {0, 5, 10, 15};
	bool totalT = true;
	bool partialT = true;
    		
    // Extract the base name without the extension
    size_t dotPos = inputFile.find_last_of('.');
    std::string baseName = inputFile.substr(0, dotPos);
    
    // Load the image from the input path
    std::string inputPath = "Input/" + folderName + "/" + inputFile;
    printf("%s\n", folderName.c_str()); // Print the folder name
        
	processImageTransforms(inputPath, baseName, threshold, lastThreshold, step, fraction, rectanglesToModify, totalT, partialT);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	printf("Temps d'exécution: %ld s\n", duration.count());

    return 0;
}
