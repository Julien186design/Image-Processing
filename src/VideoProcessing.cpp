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

    const std::string colorNuancesToString = "{" + std::to_string(colorNuances[0]) + "-" +
                std::to_string(colorNuances[1]) + "-" + std::to_string(colorNuances[2]) + "}";
    const std::string outputVideoPath = std::string(FOLDER_VIDEOS) + baseName + " - " +
        std::to_string(nFrames) + " images - " + std::to_string(fps) + " fps - " +
            colorNuancesToString + ".mp4";

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

            // Only modify pixels above threshold
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
    const double sourceFps = capture.get(cv::CAP_PROP_FPS);
    const int fps = static_cast<int>(sourceFps);

    const int startFrame = std::max(0, frames[0]);
    const int endFrame = (frames[1] <= 0 || frames[1] > totalFrames) ? totalFrames : frames[1];
    const int framesToProcess = endFrame - startFrame;

    if (framesToProcess <= 0) {
        std::cerr << "Error: invalid frame range [" << startFrame << ", " << endFrame << "]\n";
        return;
    }

    const std::string tempVideoPath = std::string(FOLDER_VIDEOS) + baseName + "_temp.mp4";
    std::string outputVideoPath = std::string(FOLDER_VIDEOS) + baseName +
        " - Edge Detector - " + std::to_string(framesToProcess) + " frames ";
    if (framesToProcess == totalFrames) {
        outputVideoPath += "- "  +  std::to_string(fps) + " fps.mp4";
    } else {
        outputVideoPath += "{" + std::to_string(frames[0]) + "-" + std::to_string(frames[1]) + "} " +
            std::to_string(fps) + " fps.mp4";
    }

    cv::VideoWriter video(tempVideoPath,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height));

    if (!video.isOpened()) {
        std::cerr << "Error: cannot create " << tempVideoPath << std::endl;
        return;
    }

    const size_t imgSize = width * height;

    // Pre-computed Gaussian kernel
    constexpr double inv16 = 1.0 / 16.0;
    constexpr double gauss[9] = {
        inv16, 2*inv16, inv16,
        2*inv16, 4*inv16, 2*inv16,
        inv16, 2*inv16, inv16
    };
    constexpr double threshold = 0.09;

    // Skip to start frame
    capture.set(cv::CAP_PROP_POS_FRAMES, startFrame);

    int frameIdx = 0;
    cv::Mat frameBGR(height, width, CV_8UC3);

    // Thread-local buffers (avoiding allocation in loop)
    std::vector<uint8_t> grayData(imgSize);
    std::vector<double> blurData(imgSize);
    std::vector<double> tx(imgSize), ty(imgSize);
    std::vector<double> gx(imgSize), gy(imgSize);
    std::vector<double> g(imgSize), theta(imgSize);
    std::vector<uint8_t> outputRGB(imgSize * 3);

    while (frameIdx < framesToProcess) {
        if (!capture.read(frameBGR)) break;

        ++frameIdx;
        std::cout << "Frame " << frameIdx << "/" << framesToProcess << "\r" << std::flush;

        // BGR → Grayscale (average) - PARALLELIZED
        #pragma omp parallel for
        for (int i = 0; i < imgSize; ++i) {
            const uint8_t* pixel = frameBGR.data + i * 3;
            grayData[i] = static_cast<uint8_t>((pixel[0] + pixel[1] + pixel[2]) / 3);
        }

        // Gaussian blur (3x3) - PARALLELIZED
        std::ranges::fill(blurData, 0.0);
        #pragma omp parallel for
        for (int r = 1; r < height - 1; ++r) {
            for (int c = 1; c < width - 1; ++c) {
                double sum = 0.0;
                for (int kr = -1; kr <= 1; ++kr) {
                    for (int kc = -1; kc <= 1; ++kc) {
                        sum += grayData[(r + kr) * width + (c + kc)] *
                               gauss[(kr + 1) * 3 + (kc + 1)];
                    }
                }
                blurData[r * width + c] = sum;
            }
        }

        // Scharr separable convolution - PARALLELIZED
        std::ranges::fill(tx, 0.0);
        std::ranges::fill(ty, 0.0);
        #pragma omp parallel for
        for (int r = 0; r < height; ++r) {
            for (uint32_t c = 1; c < width - 1; ++c) {
                const size_t idx = r * width + c;
                tx[idx] = blurData[idx + 1] - blurData[idx - 1];
                ty[idx] = 47 * blurData[idx + 1] + 162 * blurData[idx] + 47 * blurData[idx - 1];
            }
        }

        std::ranges::fill(gx, 0.0);
        std::ranges::fill(gy, 0.0);
        #pragma omp parallel for
        for (int c = 1; c < width - 1; ++c) {
            for (uint32_t r = 1; r < height - 1; ++r) {
                const size_t idx = r * width + c;
                gx[idx] = 47 * tx[idx + width] + 162 * tx[idx] + 47 * tx[idx - width];
                gy[idx] = ty[idx + width] - ty[idx - width];
            }
        }

        // Magnitude and angle - PARALLELIZED with reduction
        double mx = -INFINITY, mn = INFINITY;

        #pragma omp parallel
        {
            double local_mx = -INFINITY;
            double local_mn = INFINITY;

            #pragma omp for nowait
            for (int k = 0; k < imgSize; ++k) {
                const double x = gx[k];
                const double y = gy[k];
                g[k] = std::sqrt(x * x + y * y);
                theta[k] = std::atan2(y, x);
                local_mx = std::max(local_mx, g[k]);
                local_mn = std::min(local_mn, g[k]);
            }

            #pragma omp critical
            {
                mx = std::max(mx, local_mx);
                mn = std::min(mn, local_mn);
            }
        }

        // HSL → RGB with thresholding - PARALLELIZED
        const double range = (mx == mn) ? 1.0 : (mx - mn);
        #pragma omp parallel for
        for (int k = 0; k < imgSize; ++k) {
            const double h = theta[k] * 180.0 / M_PI + 180.0;
            const double v = ((g[k] - mn) / range > threshold) ? (g[k] - mn) / range : 0.0;
            const double s = v, l = v;

            const double c = (1 - std::abs(2 * l - 1)) * s;
            const double x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
            const double m = l - c / 2.0;

            double rt = 0, gt = 0, bt = 0;
            if (h < 60)       { rt = c; gt = x; }
            else if (h < 120) { rt = x; gt = c; }
            else if (h < 180) { gt = c; bt = x; }
            else if (h < 240) { gt = x; bt = c; }
            else if (h < 300) { bt = c; rt = x; }
            else              { bt = x; rt = c; }

            outputRGB[k * 3]     = static_cast<uint8_t>(255 * (rt + m));
            outputRGB[k * 3 + 1] = static_cast<uint8_t>(255 * (gt + m));
            outputRGB[k * 3 + 2] = static_cast<uint8_t>(255 * (bt + m));
        }

        // RGB → BGR for OpenCV - PARALLELIZED
        #pragma omp parallel for
        for (int i = 0; i < imgSize; ++i) {
            frameBGR.data[i * 3]     = outputRGB[i * 3 + 2]; // B
            frameBGR.data[i * 3 + 1] = outputRGB[i * 3 + 1]; // G
            frameBGR.data[i * 3 + 2] = outputRGB[i * 3];     // R
        }

        video.write(frameBGR);
    }

    capture.release();
    video.release();

    std::cout << "\nMerging audio with FFmpeg...\n";

    const double startTime = static_cast<double>(startFrame) / fps;
    const double duration = static_cast<double>(framesToProcess) / fps;

    std::string ffmpegCmd = "ffmpeg -i \"" + tempVideoPath + "\" -i \"" + inputVideoPath +
                            "\" -ss " + std::to_string(startTime) +
                            " -t " + std::to_string(duration) +
                            " -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -y \"" +
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
        std::cout << "Fichier MP4 détecté → edge_detector_video" << std::endl;
        edge_detector_video(baseName, inputPath, frames);
    } else {
        std::cout << "Fichier non-MP4 détecté → several_colors_transformations_streaming" << std::endl;
        several_colors_transformations_streaming(baseName, inputPath, fps, proportions, colorNuances);
    }
}
