#include "ImageProcessing.h"

#include <chrono>



int main() {
	auto start = std::chrono::high_resolution_clock::now();
    const std::string inputFile = "Oreo Cake.jpg";
    const std::string folderName = "Food";
	
	std::vector<int> thresholdsAndStep = {30, 240, 30}; // {first_Threshold, last_Threshold, step}
	int fraction = 2;
	std::vector<int> tolerance = {0, 20, 1};
	std::vector<int> rectanglesToModify = {0, 5, 10, 15};
	bool totalStepByStepT = false;
	bool totalBlackAndWhiteT = false;
	bool totalReversalT = false;
	bool partialT = true;
	bool alternatingBlackAndWhite = false; // the function is still under construction
	bool oneColor = false;
    		
    // Extract the base name without the extension
    size_t dotPos = inputFile.find_last_of('.');
    std::string baseName = inputFile.substr(0, dotPos);


    // Load the image from the input path
    std::string inputPath = "Input/" + folderName + "/" + inputFile;
    printf("%s\n", folderName.c_str()); // Print the folder name
        
	processImageTransforms(inputPath, baseName, thresholdsAndStep, fraction, rectanglesToModify, tolerance,
		totalStepByStepT, totalBlackAndWhiteT, totalReversalT,
		partialT,alternatingBlackAndWhite, oneColor);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	printf("Execution time: %ld s\n", duration.count());

    return 0;
}
