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

	const std::vector<int> thresholdsAndStep = {180, 250, 10}; // {first Threshold, last Threshold, step}
	// const std::vector<int> colorNuances = {0, 20, 40, 60, 80, 90, 110};
	const std::vector<int> colorNuances = {0, 90};
	constexpr int fraction = 4;
	// const std::vector<int> rectanglesToModify = {0, 5, 10, 15};
	const std::vector<int> rectanglesToModify = genererRectanglesInDiagonale(fraction);
	const std::vector<int> toleranceOneColor = {0, 1, 1};
	constexpr bool severalColors = false;
	constexpr bool totalBlackAndWhiteT = false;
	constexpr bool totalReversalT = false;
	constexpr bool partialT = true;
	constexpr bool alternatingBlackAndWhite = false; // the function is still under construction
	constexpr bool oneColor = false;


	processImageTransforms(inputFile, folderName, thresholdsAndStep, colorNuances, fraction,
		rectanglesToModify,	toleranceOneColor, severalColors, totalBlackAndWhiteT, totalReversalT,
		partialT,alternatingBlackAndWhite, oneColor);

	for (const auto& valeur : rectanglesToModify) {
		std::cout << valeur << " ";
	}
	std::cout << std::endl;

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);



	printf("Execution time: %ld s\n", duration.count());

	return 0;
}
