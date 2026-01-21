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
	const std::string inputFile = "Oreo Cake.jpg";
	const std::string folderName = "Food";
	
	const std::vector<int> thresholdsAndStep = {60, 250, 60}; // {first threshold, last threshold, step}
	const std::vector<int> colorNuances = {0, 120, 40}; // {first color, last color, step}
	constexpr int fraction = 2;
	std::vector<int> rectanglesToModify = range_to_vector({8, 15});
	const std::vector<int> toleranceOneColor = {0, 20, 4};
	constexpr bool severalColors = false;
	constexpr bool oneColor = true;
	constexpr bool totalBlackAndWhite = false;
	constexpr bool totalReversal = false;
	constexpr bool partial = false;
	constexpr bool partialInDiagonal = false;
	constexpr bool alternatingBlackAndWhite = false; // the function is still under construction


	processImageTransforms(inputFile, folderName, thresholdsAndStep, colorNuances, fraction,
		rectanglesToModify,	toleranceOneColor, severalColors, totalBlackAndWhite, totalReversal,
		partial, partialInDiagonal, alternatingBlackAndWhite, oneColor);

	const auto end = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	printf("Execution time: %ld s\n", duration.count());

	return 0;
}
