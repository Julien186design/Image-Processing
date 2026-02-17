#include "ImageProcessing.h"
#include "VideoProcessing.h"
#include <chrono>

int main() {
	const auto start = std::chrono::high_resolution_clock::now();
	const std::string inputFile = "Food/Oreo Cake.jpg";

	const std::vector<int> thresholdsAndStep = {60, 240, 60}; // {first threshold, last threshold, step}
	const std::vector<float> proportions = {0, 1, 0.125}; // {first proportion, last proportion, step}
	const std::vector<int> colorNuances = {0, 40, 40}; // {first color, last color, step}
	const std::vector<int> frames = {0, 0};
	constexpr int fps = 60;
	constexpr int fraction = 1;
	std::vector<int> rectanglesToModify = range_to_vector({1, 3});
	const std::vector toleranceOneColor = {0, 3, 1};
	const std::vector<float> weightOfRGB = {1, 1, 1, 1};
	constexpr bool severalColorsByProportion = false;
	constexpr bool oneColor = true;
	constexpr bool totalBlackAndWhite = false;
	constexpr bool totalReversal = false;
	constexpr bool partial = false;
	constexpr bool partialInDiagonal = false;
	constexpr bool alternatingBlackAndWhite = false; // the function is still under construction

	const auto [baseName, inputPath] = extractImageInfo(inputFile);

	processImageTransforms(baseName, inputPath, thresholdsAndStep, proportions, colorNuances,
		fraction, rectanglesToModify,	toleranceOneColor, weightOfRGB, severalColorsByProportion,
		totalBlackAndWhite, totalReversal, partial, partialInDiagonal, alternatingBlackAndWhite, oneColor);

	processVideoTransforms(baseName, inputPath, fps, proportions, colorNuances, frames);

	const auto end = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	printf("Execution time: %ld s\n", duration.count());

	return 0;
}
