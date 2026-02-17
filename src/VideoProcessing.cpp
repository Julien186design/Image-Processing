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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>

void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath,
    const int fps,
    const std::vector<float>& proportions,
    const std::vector<int>& colorNuances
) {
    // Load the input image using OpenCV
    cv::Mat baseImageMat = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (baseImageMat.empty()) {
        std::cerr << "Error: Could not load image " << inputPath << std::endl;
        return;
    }

    // Calculate the number of steps and frames
    const int numSteps = static_cast<int>((proportions[1] - proportions[0]) / proportions[2]) + 1;
    const int nFrames = numSteps * 2 * (((colorNuances[1] - colorNuances[0]) / colorNuances[2]) + 1);

    if (nFrames < 2 * fps) {
        std::cerr << "Video less than 2 seconds long, creation canceled" << std::endl;
        return;
    }

    std::cout << "Frames to process : " << nFrames << std::endl;

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

    // Parallel computation of RGB sums
    const unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    const size_t chunkSize = pixelCount / numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            const size_t start = t * chunkSize;
            const size_t end = (t == numThreads - 1) ? pixelCount : (t + 1) * chunkSize;

            for (size_t i = start; i < end; ++i) {
                const int row = static_cast<int>(i / baseImageMat.cols);
                const int col = static_cast<int>(i % baseImageMat.cols);
                const cv::Vec3b& pixel = baseImageMat.at<cv::Vec3b>(row, col);
                rgbSums[i] = pixel[0] + pixel[1] + pixel[2];
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
    threads.clear();

    // Sort RGB sums to compute thresholds
    std::vector<int> sortedRGB = rgbSums;
    std::ranges::sort(sortedRGB);

    std::vector<int> thresholds(numSteps);
    for (int i = 0; i < numSteps; ++i) {
        const float cp = proportions[0] + i * proportions[2];
        thresholds[i] = sortedRGB[static_cast<size_t>(pixelCount * cp)];
    }

    // Pre-compute pixel masks in parallel
    std::vector<std::vector<bool>> pixelMask(numSteps, std::vector<bool>(pixelCount, false));

    for (int propIdx = 0; propIdx < numSteps; ++propIdx) {
        for (unsigned int t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, propIdx, t]() {
                const size_t start = t * chunkSize;
                const size_t end = (t == numThreads - 1) ? pixelCount : (t + 1) * chunkSize;

                for (size_t pixelIdx = start; pixelIdx < end; ++pixelIdx) {
                    if (rgbSums[pixelIdx] <= thresholds[propIdx]) {
                        pixelMask[propIdx][pixelIdx] = true;
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }

    // Frame buffer and processing queue
    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
    std::condition_variable queueCV;
    std::atomic<bool> processingDone{false};
    constexpr size_t maxQueueSize = 30; // Limit memory usage

    // Video writer thread
    std::thread writerThread([&]() {
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [&]() { return !frameQueue.empty() || processingDone; });

            if (frameQueue.empty() && processingDone) {
                break;
            }

            if (!frameQueue.empty()) {
                cv::Mat frame = std::move(frameQueue.front());
                frameQueue.pop();
                lock.unlock();
                queueCV.notify_one();

                video.write(frame);
            }
        }
    });

    // Lambda function to add frame to queue
    auto enqueueFrame = [&](const cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [&]() { return frameQueue.size() < maxQueueSize; });
        frameQueue.push(frame.clone());
        lock.unlock();
        queueCV.notify_one();
    };

    // Lambda function to apply color transformation to pixels in parallel
    auto applyColorTransform = [&](cv::Mat& target, const std::vector<bool>& mask, const uint8_t newColor) {
        for (unsigned int t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, t]() {
                const size_t start = t * chunkSize;
                const size_t end = (t == numThreads - 1) ? pixelCount : (t + 1) * chunkSize;

                for (size_t pixelIdx = start; pixelIdx < end; ++pixelIdx) {
                    if (mask[pixelIdx]) {
                        const size_t row = pixelIdx / baseImageMat.cols;
                        const size_t col = pixelIdx % baseImageMat.cols;
                        target.at<cv::Vec3b>(static_cast<int>(row), static_cast<int>(col)) =
                            cv::Vec3b(newColor, newColor, newColor);
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    };

    // Lambda function to process frames with color transformations
    auto process = [&](const int propIdx, const bool reverseOrder) {
        const auto& mask = pixelMask[propIdx];
        const int startIdx = reverseOrder ? 1 : 0;
        const int reverseIdx = reverseOrder ? 0 : 1;
        cv::Mat modifiedMat;

        // Phase 1: Ascending colorNuance
        for (int colorNuance = colorNuances[0]; colorNuance <= colorNuances[1]; colorNuance += colorNuances[2]) {
            if (colorNuance == colorNuances[0]) {
                modifiedMat = baseImageMat.clone();
            }

            const uint8_t newColor = (startIdx == 1) ? colorNuance : (255 - colorNuance);
            applyColorTransform(modifiedMat, mask, newColor);
            enqueueFrame(modifiedMat);
        }

        // Phase 2: Descending colorNuance
        for (int colorNuance = colorNuances[1]; colorNuance >= colorNuances[0]; colorNuance -= colorNuances[2]) {
            const uint8_t newColor = (reverseIdx == 1) ? colorNuance : (255 - colorNuance);
            applyColorTransform(modifiedMat, mask, newColor);
            enqueueFrame(modifiedMat);
        }
    };

    // Process each step with alternating order
    bool reverseOrder = false;
    for (int i = 0; i < numSteps; ++i) {
        const float cp = proportions[0] + static_cast<float>(i) * proportions[2];
        std::cout << "Processing proportion " << cp << std::endl;
        process(i, reverseOrder);
        reverseOrder = !reverseOrder;
    }

    // Signal completion and wait for writer thread
    processingDone = true;
    queueCV.notify_one();
    writerThread.join();

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
    std::cout << "Number of frames in the original video " << totalFrames << std::endl;
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
    std::cout << "Frames to process : " << framesToProcess << std::endl;
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
