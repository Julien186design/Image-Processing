#include "ColorConfig.h"
#include "EdgeDetector.h"
#include "VideoProcessing.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <atomic>

void one_color_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::oneColor) { return; }
    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) { return;
}
    cv::Mat& baseImageMat = *imageOpt;

    const auto pipeline = OneColorPipeline::forStreaming();
    const size_t configFrames = pipeline.configCount();
    const size_t totalFrames  = configFrames * parameters::numTolerance * pipeline.passCount();

    if (totalFrames < static_cast<size_t>(parameters::fps)) {
        Logger::err("Video less than 1 second long, creation canceled");
        return;
    }
    Logger::log("Frames to process : ", totalFrames);

    std::cout << "configFrames " << configFrames << std::endl;
    std::cout << "TOLERANCE_RAM " << TOLERANCE_RAM << std::endl;

    const std::string outputVideoPath =
        OutputPathBuilder::video_one_color(baseName, totalFrames, 0);

    cv::VideoWriter writer;
    writer.open(outputVideoPath,
                cv::VideoWriter::fourcc('m','p','4','v'),
                parameters::fps,
                cv::Size(baseImageMat.cols, baseImageMat.rows));
    if (!writer.isOpened()) {
        Logger::err("Error: Could not open video writer");
        return;
    }

    const int num_threads = computeNumThreads();

    // Pre-allocate one Image per thread to avoid per-iteration heap allocation
    std::vector<Image> thread_imgs;
    thread_imgs.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_imgs.emplace_back(baseImageMat.cols, baseImageMat.rows,
                                baseImageMat.channels());
}

    for (int n = 0; n < parameters::numTolerance; ++n) {
    const int tole_pos = parameters::toleranceOneColor[1]
                       - n * parameters::toleranceOneColor[2];
    Logger::log("tole_pos ", tole_pos);

    // Allocate slots for one chunk at a time (outer: thread, inner: phase slot)
    std::vector<std::vector<std::vector<Image>>> perThreadResults(num_threads);

    for (size_t phase_base = 0; phase_base < configFrames; phase_base += TOLERANCE_RAM) {
        const size_t phase_end = std::min(phase_base + TOLERANCE_RAM, configFrames);
        const size_t chunk_size = phase_end - phase_base;

        for (int t = 0; t < num_threads; ++t)
            perThreadResults[t].assign(chunk_size, {});

        // Chunk-local chunk: ceil(chunk_size / num_threads) for ownerThread mapping
        const size_t local_chunk =
            (chunk_size + static_cast<size_t>(num_threads) - 1)
            / static_cast<size_t>(num_threads);

        #pragma omp parallel for schedule(static) num_threads(num_threads) \
        default(none) \
        shared(baseImageMat, pipeline, configFrames, tole_pos, n, \
               thread_imgs, perThreadResults, phase_base, phase_end)
        for (size_t phase = phase_base; phase < phase_end; ++phase) {
            const int tid = omp_get_thread_num();
            const size_t configIdx = phase;

            Image const& img = thread_imgs[tid];
            std::memcpy(img.data, baseImageMat.data,
                        static_cast<size_t>(baseImageMat.cols)
                        * baseImageMat.rows * baseImageMat.channels());

            // Store result at local index within this chunk
            const size_t local_phase = phase - phase_base;
            perThreadResults[tid][local_phase] = pipeline.applyStreaming(
                img, tole_pos, configIdx);
        }

        // Drain this chunk in phase order, releasing each slot immediately
        for (size_t phase = phase_base; phase < phase_end; ++phase) {
            const size_t local_phase = phase - phase_base;
            const int ownerThread = static_cast<int>(local_phase / local_chunk);

            for (const Image& result : perThreadResults[ownerThread][local_phase]) {
                const cv::Mat frame(baseImageMat.rows, baseImageMat.cols,
                                    baseImageMat.type(),
                                    const_cast<void*>(
                                        static_cast<const void*>(result.data)));
                writer.write(frame);
            }

            // Release immediately: this slot's Images are no longer needed
            std::vector<Image>().swap(perThreadResults[ownerThread][local_phase]);
        }
    }

    if (n < parameters::numTolerance - 1) {
        const int percent = (100 * (n + 1)) / parameters::numTolerance;
        sendNotification("one_color_transformations_streaming",
                         std::to_string(percent) + "% completed");
    }
}

    writer.release();
    Logger::log("\n", baseName, " --> video created");
}

void reverse_transformations_by_proportion_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::totalReversal) { return; }

    const auto imageOpt = loadImage(inputPath);
    if (!imageOpt) { return; }
    const cv::Mat& baseImageMat = *imageOpt;

    const int width    = baseImageMat.cols;
    const int height   = baseImageMat.rows;
    const int channels = baseImageMat.channels();

    // One video per reversal entry (Reversal-BT, Reversal-WT)
    for (size_t entryIdx = 0; entryIdx < reversal_step_by_step_entries.size(); ++entryIdx) {
        const auto& [suffix, output_dir] = reversal_step_by_step_entries[entryIdx];
        const bool below = reversal_below_flag(entryIdx);

        // numProportionSteps frames, one per proportion value in [proportions[0], proportions[1]]
        constexpr int nFrames = parameters::numProportionSteps;

        if constexpr (nFrames < parameters::fps) {
            Logger::err("Video less than 1 second long, creation canceled for ", suffix);
            continue;
        }

        const std::string outputPath =
            OutputPathBuilder::video_reversal(output_dir, baseName, suffix, nFrames);

        auto writerOpt = make_video_writer(outputPath, width, height, parameters::fps);
        if (!writerOpt) { continue; }
        cv::VideoWriter& writer = *writerOpt;

        // Pre-allocate one working buffer (avoids per-frame heap allocation)
        Image workImg(width, height, channels);

        for (int p_idx = 0; p_idx < nFrames; ++p_idx) {
            const float proportion =
                parameters::proportions[0] +
                static_cast<float>(p_idx) * parameters::proportions[2];

            // Skip p=0: reverse_by_proportion rejects it (sorting_pixels_by_brightness
            // returns nullopt for proportion <= 0), producing an unchanged frame.
            if (proportion <= 0.0F) { continue; }

            // Reset buffer to the original image for this frame
            std::memcpy(workImg.data, baseImageMat.data,
                        static_cast<size_t>(width) * height * channels);

            workImg.reverse_by_proportion(proportion, below);

            // Wrap Image buffer as a cv::Mat header (zero-copy)
            const cv::Mat frame(height, width,
                                baseImageMat.type(),
                                static_cast<void*>(workImg.data));
            writer.write(frame);

            Logger::logProgress(suffix, " frame ", p_idx + 1, "/", nFrames);
        }

        writer.release();
        Logger::log("\n", outputPath, " created");
    }
}

void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::complete_transformation_colors_by_proportion) {return;}
    // Load the input image using OpenCV
    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) return;
    cv::Mat& baseImageMat = *imageOpt; // référence, pas de copie

    // Calculate the number of steps and frames
    constexpr int nFrames = 2 * parameters::numProportionSteps * (
                (parameters::colorNuances[1] - parameters::colorNuances[0]) / parameters::colorNuances[2] + 1);

    if constexpr (nFrames < parameters::fps) {
        Logger::err("Video less than 1 second long, creation canceled");
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

    std::vector<int> thresholds(parameters::numProportionSteps);
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
        // const float cp = proportions.start + static_cast<float>(i) * proportions.step;
        const float cp = parameters::proportions[0] + static_cast<float>(i) * parameters::proportions[2];
        thresholds[i] = sortedRGB[static_cast<size_t>(static_cast<float>(pixelCount) * cp)];
    }

    // Pre-compute pixel masks in parallel
    std::vector pixelMask(parameters::numProportionSteps, std::vector(pixelCount, false));

    for (int propIdx = 0; propIdx < parameters::numProportionSteps; ++propIdx) {
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
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
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


void edge_detector_video(
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
    const int endFrame   = (parameters::frames[1] > totalFrames)
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

void processVideoTransforms(
    const std::string& baseName,
    const std::string& inputPath
) {
    // Checking extension MP4
    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected → edge_detector_video");
        edge_detector_video(baseName, inputPath);
    }
    else {
        Logger::log("Non-MP4 file detected → colored_transformations");
        several_colors_transformations_streaming(baseName, inputPath);
        one_color_transformations_streaming(baseName, inputPath);
        reverse_transformations_by_proportion_streaming(baseName, inputPath);
    }
}
#include "ColorConfig.h"
#include "EdgeDetector.h"
#include "VideoProcessing.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <atomic>

void one_color_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::oneColor) { return; }
    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) return;
    cv::Mat& baseImageMat = *imageOpt;

    const OneColorPipeline pipeline{};

    constexpr int num_tole =
        (parameters::toleranceOneColor[1] - parameters::toleranceOneColor[0]) / parameters::toleranceOneColor[2] + 1;

    constexpr size_t n = parameters::numProportionSteps;
    const size_t totalIterations = pipeline.configCount() * num_tole;
    const size_t totalFrames = totalIterations * n;

    if (totalFrames < static_cast<size_t>(parameters::fps)) {
        Logger::err("Video less than 1 second long, creation canceled");
        return;
    }

    Logger::log("Frames to process : ", totalFrames);

    // One VideoWriterQueue per index
    std::vector<cv::VideoWriter> writers(n);
    std::vector<std::unique_ptr<VideoWriterQueue>> queues;
    queues.reserve(n);

    for (size_t idx = 0; idx < n; ++idx) {
        const std::string outputVideoPath = OutputPathBuilder::video_one_color(baseName, totalIterations, idx);
        writers[idx].open(
            outputVideoPath,
            cv::VideoWriter::fourcc('m','p','4','v'),
            parameters::fps,
            cv::Size(baseImageMat.cols, baseImageMat.rows)
        );
        if (!writers[idx].isOpened()) {
            Logger::err("Error: Could not open video writer for index ", idx);
            return;
        }
        queues.push_back(std::make_unique<VideoWriterQueue>(writers[idx]));
    }

    ProgressNotifier notifier("One Color Video", totalFrames);
    std::atomic<std::size_t> progress{0};

    const int num_threads = computeNumThreads();

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads) \
    default(none) \
    shared(baseImageMat, pipeline, \
           totalIterations, num_tole, n, \
           queues, progress, notifier)
    for (size_t iter = 0; iter < totalIterations; ++iter)
    {
        const size_t configIdx = iter / num_tole;
        const int pos = static_cast<int>(iter % num_tole);
        const int tole_pos = (configIdx % 2 == 1) ? (num_tole - 1 - pos) : pos;

        const auto params = pipeline.buildParams(configIdx);

        Image img(baseImageMat.cols, baseImageMat.rows, baseImageMat.channels());
        std::memcpy(img.data, baseImageMat.data,
                    baseImageMat.total() * baseImageMat.channels());

        const auto results = pipeline.apply(
            img,
            tole_pos * parameters::toleranceOneColor[2] + parameters::toleranceOneColor[0],
            configIdx
        );

        for (size_t idx = 0; idx < results.size(); ++idx) {
            cv::Mat result(baseImageMat.rows, baseImageMat.cols, baseImageMat.type());
            std::memcpy(result.data, results[idx].data,
                        static_cast<size_t>(result.total() * result.channels()));
            queues[idx]->enqueue(std::move(result));
            notifier.update(++progress);
        }
    }

    for (auto& q : queues) q->finish();
    for (auto& w : writers) w.release();

    Logger::log("\n", baseName, " video created");
}

void several_colors_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::complete_transformation_colors_by_proportion) {return;}
    // Load the input image using OpenCV
    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) return;
    cv::Mat& baseImageMat = *imageOpt; // référence, pas de copie

    // Calculate the number of steps and frames
    constexpr int nFrames = 2 * parameters::numProportionSteps * (
                (parameters::colorNuances[1] - parameters::colorNuances[0]) / parameters::colorNuances[2] + 1);

    if constexpr (nFrames < parameters::fps) {
        Logger::err("Video less than 1 second long, creation canceled");
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

    std::vector<int> thresholds(parameters::numProportionSteps);
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
        // const float cp = proportions.start + static_cast<float>(i) * proportions.step;
        const float cp = parameters::proportions[0] + static_cast<float>(i) * parameters::proportions[2];
        thresholds[i] = sortedRGB[static_cast<size_t>(static_cast<float>(pixelCount) * cp)];
    }

    // Pre-compute pixel masks in parallel
    std::vector pixelMask(parameters::numProportionSteps, std::vector(pixelCount, false));

    for (int propIdx = 0; propIdx < parameters::numProportionSteps; ++propIdx) {
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
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
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


void edge_detector_video(
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
    const int endFrame   = (parameters::frames[1] > totalFrames)
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

void processVideoTransforms(
    const std::string& baseName,
    const std::string& inputPath
) {
    // Checking extension MP4
    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected → edge_detector_video");
        edge_detector_video(baseName, inputPath);
    }
    else {
        Logger::log("Non-MP4 file detected → colored_transformations");
        several_colors_transformations_streaming(baseName, inputPath);
        one_color_transformations_streaming(baseName, inputPath);
    }
}
