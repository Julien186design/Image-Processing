#include "ImageProcessing.h"

#include <chrono>
#include <omp.h>
#include <iostream>

int main() {
	#pragma omp parallel
	{
		std::cout << "Thread " << omp_get_thread_num() << " sur " << omp_get_num_threads() << std::endl;
	}
	auto start = std::chrono::high_resolution_clock::now();
	const std::string inputFile = "Food/Oreo Cake.jpg";
	
	const std::vector<int> thresholdsAndStep = {60, 180, 30}; // {first threshold, last threshold, step}
	const std::vector<float> proportions = {0, 1, 0.1};
	const std::vector<int> colorNuances = {0, 120, 10}; // {first color, last color, step}
	constexpr int fps = 25;
	constexpr int fraction = 2;
	std::vector<int> rectanglesToModify = range_to_vector({8, 15});
	const std::vector toleranceOneColor = {0, 20, 1};
	constexpr bool severalColorsByThreshold = false;
	constexpr bool severalColorsByProportion = false;
	constexpr bool oneColor = false;
	constexpr bool totalBlackAndWhite = false;
	constexpr bool totalReversal = false;
	constexpr bool partial = false;
	constexpr bool partialInDiagonal = false;
	constexpr bool alternatingBlackAndWhite = false; // the function is still under construction

	const auto [baseName, inputPath] = extractImageInfo(inputFile);

	processImageTransforms(baseName, inputPath, thresholdsAndStep, proportions, colorNuances, fraction,
		rectanglesToModify,	toleranceOneColor, severalColorsByThreshold, severalColorsByProportion,
		totalBlackAndWhite, totalReversal, partial, partialInDiagonal, alternatingBlackAndWhite, oneColor);

	
	std::cout << inputPath << std::endl;

	several_colors_transformations_streaming(baseName, inputPath,  fps, proportions, colorNuances);

	const auto end = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	printf("Execution time: %ld s\n", duration.count());

	return 0;
}
