#include "ColorConfig.h"
#include "EdgeDetector.h"
#include "VideoProcessing.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <thread>

void one_color_transformations_streaming(
    const std::string& baseName,
    const std::string& inputPath
) {
    if constexpr (!parameters::oneColor) { return; }
    auto imageOpt = loadImage(inputPath);
    if (!imageOpt) { return; }
    auto& baseImageMat = *imageOpt;

    const auto pipeline = OneColorPipeline::forStreaming();
    const size_t configFrames = pipeline.configCount();

    const size_t passCount   = pipeline.passCount();
    const size_t totalFrames =
        (configFrames + static_cast<size_t>(parameters::numTolerance) - 1)
        * passCount;

    if (totalFrames < static_cast<size_t>(parameters::fps)) {
        Logger::err(totalFrames, " frames for ",  parameters::fps, " FPS --> Video less than 1 second long, creation canceled");
        return;
    }

    Logger::log("Frames to process : ", totalFrames);
    Logger::log("configFrames ", configFrames, '\n');
    Logger::log("TOLERANCE_RAM ", TOLERANCE_RAM, '\n');

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

    // One reusable image buffer per thread to avoid per-iteration heap allocation
    std::vector<Image> thread_imgs;
    thread_imgs.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_imgs.emplace_back(baseImageMat.cols, baseImageMat.rows,
                                 baseImageMat.channels());
    }

    // ── Phase 1: all configs at max tolerance ─────────────────────────────
    for (size_t phase = 0; phase < configFrames; ++phase) {
        std::memcpy(thread_imgs.at(0).data,
                    baseImageMat.data,
                    static_cast<size_t>(baseImageMat.cols)
                    * baseImageMat.rows * baseImageMat.channels());

        thread_imgs.at(0).simplify_to_dominant_color_combinations(
            parameters::toleranceOneColor.at(1),
            &pipeline.configs.at(phase),
            pipeline.getTValues(),
            [&](Image&& result) {
                const cv::Mat frame(
                    baseImageMat.rows, baseImageMat.cols,
                    baseImageMat.type(),
                    const_cast<void*>(
                        static_cast<const void*>(result.data)));
                writer.write(frame);
                return true;
            }
        );
    }

    // ── Phase 2: last config, decreasing tolerance ────────────────────────
    constexpr int tol_min  = parameters::toleranceOneColor.at(0);
    constexpr int tol_max  = parameters::toleranceOneColor.at(1);
    constexpr int tol_step = parameters::toleranceOneColor.at(2);

    // AFTER: same pattern — one Image written and destroyed per pass.
    for (int tol = tol_max - tol_step; tol >= tol_min; tol -= tol_step) {
        const size_t configIdx = configFrames - 1;
        std::memcpy(thread_imgs.at(0).data,
                    baseImageMat.data,
                    static_cast<size_t>(baseImageMat.cols)
                    * baseImageMat.rows * baseImageMat.channels());

        thread_imgs.at(0).simplify_to_dominant_color_combinations(
            tol,
            &pipeline.configs.at(configIdx),
            pipeline.getTValues(),
            [&](Image&& result) {
                const cv::Mat frame(
                    baseImageMat.rows, baseImageMat.cols,
                    baseImageMat.type(),
                    const_cast<void*>(
                        static_cast<const void*>(result.data)));
                writer.write(frame);
                return true;
            }
        );
    }

    writer.release();
    Logger::log("\n", outputVideoPath, " --> video created");
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
    const int num_threads = computeNumThreads();

    for (size_t entryIdx = 0; entryIdx < reversal_suffixes.size(); ++entryIdx) {
        const std::string_view suffix = reversal_suffixes[entryIdx];
        const bool below = reversal_below_flag(entryIdx);

        constexpr int nFrames = parameters::numProportionSteps;

        if constexpr (nFrames < parameters::fps) {
            Logger::err("Video less than 1 second long, creation canceled for ", suffix);
            continue;
        }

        const std::string outputPath = OutputPathBuilder::video_reversal(baseName, suffix, nFrames);

        auto writerOpt = make_video_writer(outputPath, width, height, parameters::fps);
        if (!writerOpt) { continue; }
        cv::VideoWriter& writer = *writerOpt;

        // Pre-allocate one working Image per thread to avoid per-iteration heap allocation
        std::vector<Image> thread_imgs;
        thread_imgs.reserve(num_threads);
        for (int thread = 0; thread < num_threads; ++thread) {
            thread_imgs.emplace_back(width, height, channels);
        }

        // perThreadResults[tid][local_phase]: one Image per (thread, chunk-local frame slot)
        std::vector<std::vector<Image>> perThreadResults(num_threads);

        for (int phase_base = 0; phase_base < nFrames; phase_base += TOLERANCE_RAM) {
            const int phase_end  = std::min(phase_base + TOLERANCE_RAM, nFrames);
            const int chunk_size = phase_end - phase_base;

            // Resize each thread's slot vector to cover this chunk
            for (int thread = 0; thread < num_threads; ++thread) {
                perThreadResults.at(thread).assign(chunk_size, Image(width, height, channels));
            }

            // Ceiling division: determines which thread owns each local slot
            const int local_chunk =
                (chunk_size + num_threads - 1) / num_threads;

            #pragma omp parallel for schedule(static) num_threads(num_threads) \
            default(none) \
            shared(baseImageMat, thread_imgs, perThreadResults, \
                   phase_base, phase_end, below, width, height, channels)
            for (int phase = phase_base; phase < phase_end; ++phase) {
                const int tid = omp_get_thread_num();
                const int local_phase = phase - phase_base;

                const float proportion =
                    parameters::proportions.at(0) +
                    (static_cast<float>(phase) * parameters::proportions.at(2));

                // Skip p=0: reverse_by_proportion rejects proportion <= 0
                if (proportion <= 0.0F) { continue; }

                // Copy base image into this thread's working buffer
                Image& img = thread_imgs.at(tid);
                std::memcpy(img.data, baseImageMat.data,
                            static_cast<size_t>(width) * height * channels);

                img.reverse_by_proportion(proportion, below);

                // Store result in the slot owned by this thread for this chunk
                perThreadResults.at(tid).at(local_phase) = img;
            }

            // Drain chunk in sequential phase order to preserve frame ordering
            for (int phase = phase_base; phase < phase_end; ++phase) {
                const int local_phase  = phase - phase_base;
                const int ownerThread  = local_phase / local_chunk;

                const float proportion =
                    parameters::proportions.at(0) +
                    (static_cast<float>(phase) * parameters::proportions.at(2));

                // Skip frames that were not produced (proportion <= 0)
                if (proportion <= 0.0F) { continue; }

                const Image& result = perThreadResults.at(ownerThread).at(local_phase);
                const cv::Mat frame(height, width,
                                    baseImageMat.type(),
                                    const_cast<void*>(
                                        static_cast<const void*>(result.data)));
                writer.write(frame);

                // Release slot immediately after writing
                perThreadResults.at(ownerThread).at(local_phase) = Image(0, 0, 0);
            }
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
    if (!imageOpt) { return;
}
    cv::Mat& baseImageMat = *imageOpt; // référence, pas de copie

    // Calculate the number of steps and frames
    constexpr int nFrames = 2 * parameters::numProportionSteps * (
                ((parameters::colorNuances.at(1) - parameters::colorNuances.at(0)) / parameters::colorNuances.at(2)) + 1);

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

    for (unsigned int thread = 0; thread < numThreads; ++thread) {
        threads.emplace_back([&, thread]() {
            const size_t start = thread * chunkSize;
            const size_t end = (thread == numThreads - 1) ? pixelCount : (thread + 1) * chunkSize;

            for (size_t i = start; i < end; ++i) {
                const int row = static_cast<int>(i / baseImageMat.cols);
                const int col = static_cast<int>(i % baseImageMat.cols);
                const cv::Vec3b& pixel = baseImageMat.at<cv::Vec3b>(row, col);
                rgbSums.at(i) = pixel[0] + pixel[1] + pixel[2];
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
        const float cp = parameters::proportions.at(0) + (static_cast<float>(i) * parameters::proportions[2]);
        thresholds.at(i) = sortedRGB.at(std::min(
        static_cast<size_t>(static_cast<float>(pixelCount) * cp),
        pixelCount - 1
));
    }

    // Pre-compute pixel masks in parallel
    std::vector pixelMask(parameters::numProportionSteps, std::vector(pixelCount, false));

    for (int propIdx = 0; propIdx < parameters::numProportionSteps; ++propIdx) {
        for (unsigned int thread = 0; thread < numThreads; ++thread) {
            threads.emplace_back([&, propIdx, thread]() {
                const size_t start = thread * chunkSize;
                const size_t end = (thread == numThreads - 1) ? pixelCount : (thread + 1) * chunkSize;

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
        const auto& mask = pixelMask.at(propIdx);
        const int startIdx = reverseOrder ? 1 : 0;
        const int reverseIdx = reverseOrder ? 0 : 1;
        cv::Mat modifiedMat;

        // Phase 1: Ascending colorNuance
        for (int colorNuance = parameters::colorNuances.at(0); colorNuance <= parameters::colorNuances.at(1);
             colorNuance += parameters::colorNuances.at(2)) {
            if (colorNuance == parameters::colorNuances.at(0)) {
                modifiedMat = baseImageMat.clone();
            }

            const uint8_t newColor = (startIdx == 1) ? colorNuance : (255 - colorNuance);
            applyColorTransform(modifiedMat, mask, newColor);
            queue.enqueue(modifiedMat.clone());
        }

        // Phase 2: Descending colorNuance
        for (int colorNuance = parameters::colorNuances.at(1); colorNuance >= parameters::colorNuances.at(0);
             colorNuance -= parameters::colorNuances.at(2)) {
            const uint8_t newColor = (reverseIdx == 1) ? colorNuance : (255 - colorNuance);
            applyColorTransform(modifiedMat, mask, newColor);
            queue.enqueue(modifiedMat.clone());
        }
    };

    // Process each step with alternating order
    bool reverseOrder = false;
    for (int i = 0; i < parameters::numProportionSteps; ++i) {
        // const float cp = proportions.start + static_cast<float>(i) * proportions.step;
        const float cp = parameters::proportions.at(0) + (static_cast<float>(i) * parameters::proportions.at(2));
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

    constexpr int startFrame = std::max(0, parameters::frames.at(0));
    const int endFrame = (parameters::frames.at(1) == 0 || parameters::frames.at(1) > totalFrames)
                       ? totalFrames
                       : parameters::frames.at(1);
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

    const size_t imgSize = static_cast<const size_t>(width * height);
    capture.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    // Initialize pipeline ONCE before loop
    EdgeDetectorPipeline pipeline(width, height);

    int frameIdx = 0;
    cv::Mat frameBGR(height, width, CV_8UC3);
    std::vector<uint8_t> grayData(imgSize);

    while (frameIdx < framesToProcess) {
        if (!capture.read(frameBGR)) { break;
}

        ++frameIdx;
        Logger::logProgress("Frame ", frameIdx, "/", framesToProcess);

        // BGR -> grayscale: average of B, G, R channels
        #pragma omp parallel for default(none) shared(imgSize, frameBGR, grayData)
            for (int i = 0; i < imgSize; ++i) {
                const uint8_t* pixel = frameBGR.data + (i * 3);
                grayData.at(i) = static_cast<uint8_t>((pixel[0] + pixel[1] + pixel[2]) / 3);
            }

        // Process frame through edge detection pipeline (zero allocation)
        const std::vector<uint8_t>& rgb = pipeline.process(grayData.data());

        // RGB -> BGR conversion for OpenCV compatibility
        #pragma omp parallel for default(none) shared(imgSize, frameBGR, rgb)
            for (int i = 0; i < imgSize; ++i) {
                frameBGR.data[static_cast<ptrdiff_t>(i * 3)]     = rgb.at((i * 3) + 2); // B
                frameBGR.data[(i * 3) + 1] = rgb.at((i * 3) + 1); // G
                frameBGR.data[(i * 3) + 2] = rgb.at(i * 3);     // R
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
