#ifndef IMAGE_PROCESSING_VIDEOCREATION_H
#define IMAGE_PROCESSING_VIDEOCREATION_H

#include "ImageProcessing.h"
#include "VideoProcessing.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <atomic>


inline void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    // Load the input image using OpenCV
    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) return;
    cv::Mat& baseImageMat = *imageOpt; // référence, pas de copie

    // Calculate the number of steps and frames
    // const int numSteps = static_cast<int>((proportions.stop - proportions.start) / proportions.step) + 1;
    constexpr int numSteps =
        static_cast<int>((parameters::proportions[1] - parameters::proportions[0]) / parameters::proportions[2]) + 1;
    constexpr int nFrames =
        numSteps * 2 * ((parameters::colorNuances[1] - parameters::colorNuances[0]) / parameters::colorNuances[2] + 1);

    if constexpr (nFrames < parameters::fps) {
        Logger::err("Video less than 1 seconds long, creation canceled");
        return;
    }

    Logger::log("Frames to process : ", nFrames);

    // Generate output video path
    const std::string outputVideoPath = OutputPathBuilder::video_several_colors(
        baseName, nFrames);

    // Initialize video writer
    cv::VideoWriter video(outputVideoPath,
                          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          parameters::fps,
                          cv::Size(baseImageMat.cols, baseImageMat.rows));

    if (!video.isOpened()) {
        Logger::err("Error: Could not open video writer for ", outputVideoPath);
        return;
    }

    // Pre-compute RGB sums for thresholding
    const size_t pixelCount = baseImageMat.rows * baseImageMat.cols;
    std::vector rgbSums(pixelCount, 0);

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
        // const float cp = proportions.start + static_cast<float>(i) * proportions.step;
        const float cp = parameters::proportions[0] + static_cast<float>(i) * parameters::proportions[2];
        thresholds[i] = sortedRGB[static_cast<size_t>(static_cast<float>(pixelCount) * cp)];
    }

    // Pre-compute pixel masks in parallel
    std::vector pixelMask(numSteps, std::vector(pixelCount, false));

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
    VideoWriterQueue queue(video);

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
        for (int colorNuance = parameters::colorNuances[0]; colorNuance <= parameters::colorNuances[1];
             colorNuance += parameters::colorNuances[2]) {
            if (colorNuance == parameters::colorNuances[0]) {
                modifiedMat = baseImageMat.clone();
            }

            const uint8_t newColor = (startIdx == 1) ? colorNuance : (255 - colorNuance);
            applyColorTransform(modifiedMat, mask, newColor);
            queue.enqueue(modifiedMat.clone());
        }

        // Phase 2: Descending colorNuance
        for (int colorNuance = parameters::colorNuances[1]; colorNuance >= parameters::colorNuances[0];
             colorNuance -= parameters::colorNuances[2]) {
            const uint8_t newColor = (reverseIdx == 1) ? colorNuance : (255 - colorNuance);
            applyColorTransform(modifiedMat, mask, newColor);
            queue.enqueue(modifiedMat.clone());
        }
    };

    // Process each step with alternating order
    bool reverseOrder = false;
    for (int i = 0; i < numSteps; ++i) {
        // const float cp = proportions.start + static_cast<float>(i) * proportions.step;
        const float cp = parameters::proportions[0] + static_cast<float>(i) * parameters::proportions[2];
        Logger::log("Processing proportion ", cp);
        process(i, reverseOrder);
        reverseOrder = !reverseOrder;
    }

    // Signal completion and wait for writer thread
    queue.finish();

    // Release resources
    video.release();
    Logger::log("\n", outputVideoPath, " created");
}

inline void one_color_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::oneColor) {return;}
    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) return;
    cv::Mat& baseImageMat = *imageOpt;

    const OneColorPipeline pipeline{};

    constexpr auto& tol = parameters::toleranceOneColor;

    constexpr int num_tole =
        (tol[1] - tol[0]) / tol[2] + 1;

    const size_t totalIterations =
        pipeline.configCount() * num_tole;

    if (totalIterations < static_cast<size_t>(parameters::fps)) {
        Logger::err("Video less than 1 seconds long, creation canceled");
        return;
    }

    Logger::log("Frames to process : ", totalIterations);

    const std::string outputVideoPath = OutputPathBuilder::video_one_color(baseName, totalIterations);

    cv::VideoWriter video(
        outputVideoPath,
        cv::VideoWriter::fourcc('m','p','4','v'),
        parameters::fps,
        cv::Size(baseImageMat.cols, baseImageMat.rows)
    );

    if (!video.isOpened()) {
        Logger::err("Error: Could not open video writer");
        return;
    }

    VideoWriterQueue queue(video);
    ProgressNotifier notifier("One Color Video", totalIterations);
    std::atomic<std::size_t> progress{0};

    const int num_threads = computeNumThreads();

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads) \
    default(none) \
    shared(baseImageMat, pipeline, \
           totalIterations, num_tole, \
           queue, progress, notifier)
    for (size_t iter = 0; iter < totalIterations; ++iter)
    {
        const size_t configIdx = iter / num_tole;

        const int tole =
            tol[0] + static_cast<int>(iter % num_tole) * tol[2];

        const auto params = pipeline.buildParams(configIdx);

        Image img(baseImageMat.cols,
                  baseImageMat.rows,
                  baseImageMat.channels());

        std::memcpy(img.data,
                    baseImageMat.data,
                    static_cast<size_t>(
                        baseImageMat.total() *
                        baseImageMat.channels()));

        pipeline.apply(img, tole, configIdx, params);

        cv::Mat result(baseImageMat.rows,
                       baseImageMat.cols,
                       baseImageMat.type());

        std::memcpy(result.data,
                    img.data,
                    static_cast<size_t>(
                        result.total() *
                        result.channels()));

        queue.enqueue(std::move(result));
        notifier.update(++progress);
    }

    queue.finish();
    video.release();

    Logger::log("\n", outputVideoPath, " created");
}

inline void edge_detector_video(
    const std::string& baseName,
    const std::string& inputVideoPath
) {
    cv::VideoCapture capture(inputVideoPath);

    if (!capture.isOpened()) {
        Logger::err("Error: cannot open ", inputVideoPath);
        return;
    }

    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    Logger::log("Number of frames in the original video : ", totalFrames);
    const double fps_edge_detector_video = capture.get(cv::CAP_PROP_FPS);

    constexpr int startFrame = std::max(0, parameters::frames[0]);
    const int endFrame   = (parameters::frames[1] <= 0 || parameters::frames[1] > totalFrames)
                         ? totalFrames
                         : parameters::frames[1];
    const int framesToProcess = endFrame - startFrame;

    if (framesToProcess <= 0) {
        Logger::err("Error: invalid frame range [", startFrame, ", ", endFrame, "]");
        return;
    }
    Logger::log("Frames to process : ", framesToProcess);
    const std::string tempVideoPath = OutputPathBuilder::video_edge_detector_temp(baseName);

    cv::VideoWriter video(tempVideoPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps_edge_detector_video,
        cv::Size(width, height));

    if (!video.isOpened()) {
        Logger::err("Error: cannot create ", tempVideoPath);
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
        Logger::logProgress("Frame ", frameIdx, "/", framesToProcess);

        // BGR -> grayscale: average of B, G, R channels
        #pragma omp parallel for default(none) shared(imgSize, frameBGR, grayData)
            for (int i = 0; i < imgSize; ++i) {
                const uint8_t* pixel = frameBGR.data + i * 3;
                grayData[i] = static_cast<uint8_t>((pixel[0] + pixel[1] + pixel[2]) / 3);
            }

        // Process frame through edge detection pipeline (zero allocation)
        const std::vector<uint8_t>& rgb = pipeline.process(grayData.data());

        // RGB -> BGR conversion for OpenCV compatibility
        #pragma omp parallel for default(none) shared(imgSize, frameBGR, rgb)
            for (int i = 0; i < imgSize; ++i) {
                frameBGR.data[i * 3]     = rgb[i * 3 + 2]; // B
                frameBGR.data[i * 3 + 1] = rgb[i * 3 + 1]; // G
                frameBGR.data[i * 3 + 2] = rgb[i * 3];     // R
            }

        video.write(frameBGR);
    }

    capture.release();
    video.release();

    Logger::log("\nMerging audio with FFmpeg...");

    const double startTime = static_cast<double>(startFrame) / fps_edge_detector_video;
    const double duration = static_cast<double>(framesToProcess) / fps_edge_detector_video;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << fps_edge_detector_video;

    std::string outputVideoPath = OutputPathBuilder::video_edge_detector(
        baseName, framesToProcess, totalFrames, fps_edge_detector_video);

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

    if (const int result = system(ffmpegCmd.c_str()); result == 0) {
        std::remove(tempVideoPath.c_str());
        Logger::log(outputVideoPath, " successfully created with audio");
    } else {
        Logger::err("Warning: FFmpeg failed. Video saved without audio: ", tempVideoPath);
    }
}

#endif //IMAGE_PROCESSING_VIDEOCREATION_H
