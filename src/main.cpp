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
	
	const std::vector<int> thresholdsAndStep = {30, 250, 30}; // {first_Threshold, last_Threshold, step}
	const std::vector<int> colorNuances = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
	constexpr int fraction = 2;
	const std::vector<int> tolerance = {0, 1, 1};
	const std::vector<int> rectanglesToModify = {0, 5, 10, 15};
	constexpr bool severalColors = true;
	constexpr bool totalBlackAndWhiteT = false;
	constexpr bool totalReversalT = false;
	constexpr bool partialT = false;
	constexpr bool alternatingBlackAndWhite = false; // the function is still under construction
	constexpr bool oneColor = false;


	processImageTransforms(inputFile, folderName, thresholdsAndStep, colorNuances, fraction,
		rectanglesToModify,	tolerance, severalColors, totalBlackAndWhiteT, totalReversalT,
		partialT,alternatingBlackAndWhite, oneColor);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	printf("Execution time: %ld s\n", duration.count());

	return 0;
}
