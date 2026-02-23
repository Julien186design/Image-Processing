#include "ImageProcessing.h"
#include "ProcessingConfig.h"
#include <chrono>
#include <cstdlib> // Required for system()

int main() {
	const auto start = std::chrono::high_resolution_clock::now();
	const std::string inputFile = "Food/Oreo Cake.jpg";
    const auto [baseName, inputPath] = extractImageInfo(inputFile);

    image_and_video_processing(baseName, inputPath);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    printf("Execution time: %ld s\n", duration.count());

    // Send a desktop notification when the program finishes
    (void)system(("notify-send 'Program Execution Complete' '" +
        baseName + " - The image and video processing has finished.'").c_str());

    return 0;
}
