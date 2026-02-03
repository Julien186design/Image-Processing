#include "VideoProcessing.h"
#include "TransformationsConfig.h"
#include "Image.h"
#include "ImageProcessing.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath,
    const int fps,
    const std::vector<float>& proportions,
    const std::vector<int>& colorNuances
) {
    // Load the input image using OpenCV to avoid manual data handling
    cv::Mat baseImageMat = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (baseImageMat.empty()) {
        std::cerr << "Error: Could not load image " << inputPath << std::endl;
        return;
    }


    // Initialize modified image as a copy of the base image
    cv::Mat modifiedMat = baseImageMat.clone();

    // Calculate the number of steps and frames
    const int numSteps = static_cast<int>((proportions[1] - proportions[0]) / proportions[2]) + 1;
    const int nFrames = numSteps * 2 * (((colorNuances[1] - colorNuances[0]) / colorNuances[2]) + 1);

    // Generate output video path
    const std::string colorNuancesToString = "{" + std::to_string(colorNuances[0]) + "-" +
                                              std::to_string(colorNuances[1]) + "-" +
                                              std::to_string(colorNuances[2]) + "}";
    const std::string outputVideoPath = std::string(FOLDER_VIDEOS) + baseName + " - " +
                                        std::to_string(nFrames) + " images - " +
                                        std::to_string(fps) + " fps - " +
                                        colorNuancesToString + ".mp4";

    // Initialize video writer
    cv::VideoWriter video(outputVideoPath,
                          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          fps,
                          cv::Size(baseImageMat.cols, baseImageMat.rows));

    if (!video.isOpened()) {
        std::cerr << "Error: Could not open video writer for " << outputVideoPath << std::endl;
        return;
    }

    // Pre-compute RGB sums for thresholding
    const size_t pixelCount = baseImageMat.rows * baseImageMat.cols;
    std::vector<int> rgbSums(pixelCount, 0);

    for (size_t i = 0; i < pixelCount; ++i) {
        const cv::Vec3b& pixel = baseImageMat.at<cv::Vec3b>(i / baseImageMat.cols, i % baseImageMat.cols);
        rgbSums[i] = pixel[0] + pixel[1] + pixel[2];
    }

    // Sort RGB sums to compute thresholds
    std::vector<int> sortedRGB = rgbSums;
    std::sort(sortedRGB.begin(), sortedRGB.end());

    std::vector<int> thresholds(numSteps);
    for (int i = 0; i < numSteps; ++i) {
        const float cp = proportions[0] + i * proportions[2];
        thresholds[i] = sortedRGB[static_cast<size_t>(pixelCount * cp)];
    }

    // Pre-compute pixel masks to avoid repeated calculations
    std::vector<std::vector<bool>> pixelMask(numSteps, std::vector<bool>(pixelCount, false));
    for (int propIdx = 0; propIdx < numSteps; ++propIdx) {
        for (size_t pixelIdx = 0; pixelIdx < pixelCount; ++pixelIdx) {
            if (rgbSums[pixelIdx] <= thresholds[propIdx]) {
                pixelMask[propIdx][pixelIdx] = true;
            }
        }
    }

    // Lambda function to write a frame to the video
    auto writeFrame = [&]() {
        video.write(modifiedMat);
    };

    // Lambda function to process a frame with color transformations
    auto process = [&](const int propIdx, const bool reverseOrder) {
        const auto& mask = pixelMask[propIdx];
        const int startIdx = reverseOrder ? 1 : 0;
        const int reverseIdx = reverseOrder ? 0 : 1;

        // Phase 1: Ascending colorNuance
        for (int colorNuance = colorNuances[0]; colorNuance <= colorNuances[1]; colorNuance += colorNuances[2]) {
            if (colorNuance == colorNuances[0]) {
                modifiedMat = baseImageMat.clone();
            }

            const uint8_t newColor = (startIdx == 1) ? colorNuance : (255 - colorNuance);

            for (size_t pixelIdx = 0; pixelIdx < pixelCount; ++pixelIdx) {
                if (mask[pixelIdx]) {
                    const int row = pixelIdx / baseImageMat.cols;
                    const int col = pixelIdx % baseImageMat.cols;
                    modifiedMat.at<cv::Vec3b>(row, col) = cv::Vec3b(newColor, newColor, newColor);
                }
            }
            writeFrame();
        }

        // Phase 2: Descending colorNuance
        for (int colorNuance = colorNuances[1]; colorNuance >= colorNuances[0]; colorNuance -= colorNuances[2]) {
            const uint8_t newColor = (reverseIdx == 1) ? colorNuance : (255 - colorNuance);

            for (size_t pixelIdx = 0; pixelIdx < pixelCount; ++pixelIdx) {
                if (mask[pixelIdx]) {
                    const int row = pixelIdx / baseImageMat.cols;
                    const int col = pixelIdx % baseImageMat.cols;
                    modifiedMat.at<cv::Vec3b>(row, col) = cv::Vec3b(newColor, newColor, newColor);
                }
            }
            writeFrame();
        }
    };

    // Process each step with alternating order
    bool reverseOrder = false;
    for (int i = 0; i < numSteps; ++i) {
        const float cp = proportions[0] + i * proportions[2];
        std::cout << "Processing proportion " << cp << std::endl;
        process(i, reverseOrder);
        reverseOrder = !reverseOrder;
    }

    // Release resources
    video.release();
    std::cout << "\n" << outputVideoPath << " created\n";
}


void edge_detector_video(
    const std::string& baseName,
    const std::string& inputVideoPath,
    const std::vector<int>& frames
) {
    cv::VideoCapture capture(inputVideoPath);

    if (!capture.isOpened()) {
        std::cerr << "Error: cannot open " << inputVideoPath << std::endl;
        return;
    }

    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "totalFrames " << totalFrames << std::endl;
    const double fps = capture.get(cv::CAP_PROP_FPS);

    const int startFrame = std::max(0, frames[0]);
    const int endFrame   = (frames[1] <= 0 || frames[1] > totalFrames)
                         ? totalFrames
                         : frames[1];
    const int framesToProcess = endFrame - startFrame;

    if (framesToProcess <= 0) {
        std::cerr << "Error: invalid frame range [" << startFrame << ", " << endFrame << "]\n";
        return;
    }

    const std::string tempVideoPath = std::string(FOLDER_VIDEOS) + baseName + "_temp.mp4";

    cv::VideoWriter video(tempVideoPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height));

    if (!video.isOpened()) {
        std::cerr << "Error: cannot create " << tempVideoPath << std::endl;
        return;
    }

    const size_t imgSize = width * height;
    capture.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    // Initialize pipeline ONCE before loop
    EdgeDetectorPipeline pipeline(width, height, 0.09);

    int frameIdx = 0;
    cv::Mat frameBGR(height, width, CV_8UC3);
    std::vector<uint8_t> grayData(imgSize);

    while (frameIdx < framesToProcess) {
        if (!capture.read(frameBGR)) break;

        ++frameIdx;
        std::cout << "Frame " << frameIdx << "/" << framesToProcess << "\r" << std::flush;

        // BGR → Grayscale
        #pragma omp parallel for
        for (int i = 0; i < imgSize; ++i) {
            const uint8_t* pixel = frameBGR.data + i * 3;
            grayData[i] = static_cast<uint8_t>((pixel[0] + pixel[1] + pixel[2]) / 3);
        }

        // Process with pipeline (ZERO allocation)
        const std::vector<uint8_t>& rgb = pipeline.process(grayData.data());

        // RGB → BGR for OpenCV
        #pragma omp parallel for
        for (int i = 0; i < imgSize; ++i) {
            frameBGR.data[i * 3]     = rgb[i * 3 + 2]; // B
            frameBGR.data[i * 3 + 1] = rgb[i * 3 + 1]; // G
            frameBGR.data[i * 3 + 2] = rgb[i * 3];     // R
        }

        video.write(frameBGR);
    }

    capture.release();
    video.release();

    std::cout << "\nMerging audio with FFmpeg...\n";

    const double startTime = static_cast<double>(startFrame) / fps;
    const double duration = static_cast<double>(framesToProcess) / fps;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << fps;

    std::string outputVideoPath = std::string(FOLDER_VIDEOS) + baseName +
        " - Edge Detector - " + std::to_string(framesToProcess) + " frames ";
    if (framesToProcess == totalFrames) {
        outputVideoPath += "- "  +  oss.str() + " fps.mp4";
    } else {
        outputVideoPath += "{" + std::to_string(frames[0]) + "-" + std::to_string(frames[1]) + "} " +
            oss.str() + " fps.mp4";
    }
    std::string ffmpegCmd =
        "ffmpeg -fflags +genpts "
        "-i \"" + tempVideoPath + "\" "
        "-ss " + std::to_string(startTime) + " "
        "-i \"" + inputVideoPath + "\" "
        "-t " + std::to_string(duration) + " "
        "-map 0:v:0 -map 1:a:0? "
        "-c:v copy -c:a aac "
        "-avoid_negative_ts make_zero "
        "-shortest -y \"" +
        outputVideoPath + "\" 2>&1";

    const int result = system(ffmpegCmd.c_str());

    if (result == 0) {
        std::remove(tempVideoPath.c_str());
        std::cout << outputVideoPath << " successfully created with audio\n";
    } else {
        std::cerr << "Warning: FFmpeg failed. Video saved without audio: " << tempVideoPath << "\n";
    }
}


void processVideoTransforms(
    const std::string& baseName,
    const std::string& inputPath,
    const int fps,
    const std::vector<float>& proportions,
    const std::vector<int>& colorNuances,
    const std::vector<int>& frames
) {
    // Checking extension MP4
    if (inputPath.length() >= 4 && [&]() {
        const std::string ext = inputPath.substr(inputPath.length() - 4);
        return ext == ".mp4" || ext == ".MP4";
    }()) {
        std::cout << "MP4 file detected → edge_detector_video" << std::endl;
        edge_detector_video(baseName, inputPath, frames);
    } else {
        std::cout << "Non-MP4 file detected → several_colors_transformations_streaming" << std::endl;
        several_colors_transformations_streaming(baseName, inputPath, fps, proportions, colorNuances);
    }
}
