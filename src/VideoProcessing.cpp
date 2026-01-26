#include "VideoProcessing.h"
#include "TransformationsConfig.h"
#include "Image.h"
#include "ImageProcessing.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>



void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath,
    const int fps,
    const std::vector<float>& proportions,
    const std::vector<int>& colorNuances
) {
    const Image baseImage(inputPath.c_str());

    // Pre-convert base image to BGR format (OpenCV native)
    std::vector<uint8_t> baseImageBGR(baseImage.size);
    for(size_t i = 0; i < baseImage.size; i += 3) {
        baseImageBGR[i]     = baseImage.data[i + 2]; // B
        baseImageBGR[i + 1] = baseImage.data[i + 1]; // G
        baseImageBGR[i + 2] = baseImage.data[i];     // R
    }

    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
    std::vector<uint8_t> modifiedBGR(baseImage.size);

    const int numSteps = static_cast<int>((proportions[1] - proportions[0]) / proportions[2]) + 1;
    const int nFrames = numSteps * 2 * (((colorNuances[1] - colorNuances[0]) / colorNuances[2]) + 1);

    const std::string outputVideoPath = std::string(FOLDER_VIDEOS) + baseName + " - " +
        std::to_string(nFrames) + " images - " + std::to_string(fps) + " fps.mp4";

    cv::VideoWriter video(outputVideoPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(baseImage.w, baseImage.h));

    // Pre-allocate cv::Mat wrapper (no data copy)
    cv::Mat frameBGR(baseImage.h, baseImage.w, CV_8UC3, modifiedBGR.data());

    // ========== PRE-COMPUTE THRESHOLDS ==========
    const size_t pixelCount = baseImage.w * baseImage.h;
    std::vector<int> rgbSums(pixelCount);

    for(size_t i = 0, idx = 0; i < baseImage.size; i += baseImage.channels, ++idx) {
        rgbSums[idx] = baseImage.data[i] + baseImage.data[i+1] + baseImage.data[i+2];
    }

    std::vector<int> thresholds(numSteps);
    std::vector<int> sortedRGB = rgbSums;
    std::ranges::sort(sortedRGB);

    for(int i = 0; i < numSteps; ++i) {
        const float cp = proportions[0] + i * proportions[2];
        thresholds[i] = sortedRGB[static_cast<size_t>(pixelCount * cp)];
    }

    // ========== OPTIMIZED FRAME WRITER ==========
    auto writeFrame = [&]() {
        // Convert RGB to BGR in-place
        for(size_t i = 0; i < baseImage.size; i += 3) {
            modifiedBGR[i]     = modified.get().data[i + 2]; // B
            modifiedBGR[i + 1] = modified.get().data[i + 1]; // G
            modifiedBGR[i + 2] = modified.get().data[i];     // R
        }
        video.write(frameBGR); // frameBGR already points to modifiedBGR
    };

    // ========== PRE-COMPUTE PIXEL MASKS ==========
    std::vector<std::vector<size_t>> pixelsBelowThreshold(numSteps);
    for(int propIdx = 0; propIdx < numSteps; ++propIdx) {
        pixelsBelowThreshold[propIdx].reserve(pixelCount / 2); // Estimate
        for(size_t pixelIdx = 0; pixelIdx < pixelCount; ++pixelIdx) {
            if(rgbSums[pixelIdx] <= thresholds[propIdx]) {
                pixelsBelowThreshold[propIdx].push_back(pixelIdx * baseImage.channels);
            }
        }
    }

    // ========== OPTIMIZED PROCESSING ==========
    auto process = [&](const int propIdx, const bool reverseOrder) {
        const auto& pixelIndices = pixelsBelowThreshold[propIdx];
        const int startIdx = reverseOrder ? 1 : 0;
        const int reverseIdx = reverseOrder ? 0 : 1;

        // Phase 1: ascending colorNuance
        for(int colorNuance = colorNuances[0];
            colorNuance <= colorNuances[1];
            colorNuance += colorNuances[2])
        {
            if(colorNuance == colorNuances[0]) {
                modified.resetFrom(baseImage);
            }

            const uint8_t newColor = (startIdx == 1) ? colorNuance : (255 - colorNuance);

            // Only modify pixels below threshold
            for(const size_t byteIdx : pixelIndices) {
                modified.get().data[byteIdx] =
                modified.get().data[byteIdx + 1] =
                modified.get().data[byteIdx + 2] = newColor;
            }
            writeFrame();
        }

        // Phase 2: descending colorNuance
        for(int colorNuance = colorNuances[1];
            colorNuance >= colorNuances[0];
            colorNuance -= colorNuances[2])
        {
            const uint8_t newColor = (reverseIdx == 1) ? colorNuance : (255 - colorNuance);

            for(const size_t byteIdx : pixelIndices) {
                modified.get().data[byteIdx] =
                modified.get().data[byteIdx + 1] =
                modified.get().data[byteIdx + 2] = newColor;
            }
            writeFrame();
        }
    };

    bool reverseOrder = false;
    for(int i = 0; i < numSteps; ++i) {
        const float cp = proportions[0] + i * proportions[2];
        std::cout << "Processing proportion " << cp << std::endl;
        process(i, reverseOrder);
        reverseOrder = !reverseOrder;
    }

    video.release();
    std::cout << "\n" << outputVideoPath << " created\n";
}


