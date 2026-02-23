#include "Image.h"
#include "ImageProcessing.h"
#include "TransformationsConfig.h"
#include <vector>
#include <functional>
#include <string>
#include <cstring>
#include <immintrin.h> // For AVX2 intrinsics
#include <format>
#include <execution>
#include <future>
#include <algorithm>

using GenericTransformationFunc = std::function<void(Image&, int, const std::vector<int>&)>;


void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& tolerance,
    const std::vector<float>& weightOfRGB,
    const bool average
) {
    const std::string ONE_COLOR_FOLDER = std::string(OUTPUT_FOLDER) + "One Color/";
    const std::string prefix0 = ONE_COLOR_FOLDER + baseName + " - Average 0 - Tolerance ";
    const std::string prefix1 = ONE_COLOR_FOLDER + baseName + " - Average 1 - Tolerance ";
    const std::string suffix  = ".png";

    const OneColorPipeline pipeline(weightOfRGB, average);

    const int num_tole        = (tolerance[1] - tolerance[0]) / tolerance[2] + 1;
    const size_t total_iterations = pipeline.configCount() * num_tole;

    #pragma omp parallel default(none) \
    shared(baseImage, pipeline, tolerance, prefix0, prefix1, suffix, num_tole, total_iterations)
    {
        ImageBuffer buffer(baseImage.w, baseImage.h, baseImage.channels);
        std::string path;
        path.reserve(256);

    #pragma omp for schedule(dynamic)
        for (size_t iter = 0; iter < total_iterations; ++iter) {
            const size_t config_idx = iter / num_tole;
            const int    tole       = tolerance[0] + (iter % num_tole) * tolerance[2];

            const auto p = pipeline.buildParams(config_idx);

            buffer.resetFrom(baseImage);
            OneColorPipeline::applyWithoutAverage(buffer.get(), tole, p);

            path = prefix0 + std::to_string(tole) + p.weightedColors + suffix;
            buffer.saveAs(path.c_str());

            if (pipeline.average) {
                buffer.resetFrom(baseImage);
                OneColorPipeline::applyWithAverage(
                    buffer.get(), tole, pipeline.configs[config_idx]);

                path = prefix1 + std::to_string(tole) + p.weightedColors + suffix;
                buffer.saveAs(path.c_str());
            }
        }
    }
}

void diffmapTransformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& tolerance,
    const std::vector<float>& weightOfRGB
) {

	const std::string ONE_COLOR_FOLDER = std::string(OUTPUT_FOLDER) + "One Color/";
	const std::string DIFFMAP_FOLDER = std::string(OUTPUT_FOLDER) + "Diffmap/";

	ImageBuffer buffer(baseImage.w, baseImage.h, baseImage.channels);

	const std::string prefix0 = ONE_COLOR_FOLDER + baseName + " - Average 0 - Tolerance ";
	const std::string prefix1 = ONE_COLOR_FOLDER + baseName + " - Average 1 - Tolerance ";
	const std::string weightedColors = makeWeightedColors(weightOfRGB);
	const std::string suffix = ".png";


    for (int i = tolerance[0]; i <= tolerance[1]; i += tolerance[2]) {

        std::string path1;
        path1.reserve(256);
        path1 = ONE_COLOR_FOLDER;
        path1 += baseName;
        path1 += " - Average 1 - Tolerance ";
        path1 += std::to_string(i);
        path1 += ".png";

        std::string path2;
        path2.reserve(256);
        path2 = ONE_COLOR_FOLDER;
        path2 += baseName;
        path2 += " - Average 0 - Tolerance ";
        path2 += std::to_string(i);
        path2 += ".png";

        Image image1(path1.c_str());
        Image image2(path2.c_str());

        ImageBuffer diffBuffer(image1.w, image1.h, image1.channels);
        diffBuffer.resetFrom(image1);
        diffBuffer.get().diffmap(image2);

        std::string outputPath;
        outputPath.reserve(256);
        outputPath = DIFFMAP_FOLDER;
        outputPath += baseName;
        outputPath += " - Diffmap - Tolerance ";
        outputPath += std::to_string(i);
        outputPath += ".png";

        diffBuffer.saveAs(outputPath.c_str());
    }
}


void several_colors_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<float>& proportions,
    const std::vector<int>& colorNuances
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    constexpr int percentInt50 = 50;
    const std::string percent50Str = std::to_string(percentInt50);

    auto saveImage = [&](const size_t transformIdx, const int percentInt, const int colorNuance) {
        const std::string percentStr = std::to_string(percentInt);
        const std::string nuanceStr = std::to_string(colorNuance);

        std::string outputPath = OutputPathBuilder::buildStandard(
            total_step_by_step_output_dirs[transformIdx],
            baseName,
            " - ",
            total_step_by_step_suffixes[transformIdx],
            percentInt
        );
        outputPath += "% - CN " + nuanceStr + ".png";
        modified.saveAs(outputPath.c_str());
    };

    auto processTransformations = [&](const float proportion, const int percentInt, const bool saveTo120) {
        for (size_t transformIdx = 0; transformIdx < transformation_params.size(); ++transformIdx) {
            const auto [below, dark] = transformation_params[transformIdx];

            for (int colorNuance = colorNuances[0];
                 colorNuance <= colorNuances[1];
                 colorNuance += colorNuances[2]) {

                modified.resetFrom(baseImage);
                modified.get().threshold_by_proportion(proportion, colorNuance, dark, below);

                saveImage(transformIdx, percentInt, colorNuance);

                if (saveTo120) {
                    std::string path120 = OutputPathBuilder::build120(
                        FOLDER_120,
                        baseName,
                        " - ",
                        total_step_by_step_suffixes[transformIdx]
                    );
                    path120 += " 50% - CN " + std::to_string(colorNuance) + ".png";
                    modified.saveAs(path120.c_str());
                }
            }
        }
    };

    processTransformations(0.5f, percentInt50, true);

    for (float p = proportions[0]; p <= proportions[1] + 1e-6f; p += proportions[2]) {
        if (std::abs(p - 0.5f) > 0.001f) {
            processTransformations(p, static_cast<int>(p * 100), false);
        }
    }
}


void partial_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<float>& proportions,
    std::vector<int>& rectangles,
    const std::vector<int>& colorNuances,
    const int fraction
) {
    const std::vector<std::string> partialDirs = generatePartialOutputDirs();
    const std::vector<std::string> partialSuffixes = generatePartialSuffixes();

    bool diagonal = false;
    if (!rectangles.empty() && rectangles[0] == -1) {
        rectangles.erase(rectangles.begin());
        diagonal = true;
    }

    const int num_threads = computeNumThreads();

    // Precompute proportion values
    std::vector<float> prop_values;
    prop_values.reserve(
        static_cast<size_t>(
            (proportions[1] - proportions[0]) / proportions[2] + 2
        )
    );

    for (float p = proportions[0];
         p <= proportions[1] + 1e-6f;
         p += proportions[2]) {
        prop_values.push_back(p);
    }

    const size_t num_props = prop_values.size();
    const size_t num_transforms = partial_transformations_by_proportion_func.size();

    #pragma omp parallel for collapse(3) schedule(dynamic) num_threads(num_threads) \
    default(none) \
    shared(prop_values, partial_transformations_by_proportion_func, colorNuances, \
    partialDirs, partialSuffixes, baseImage, baseName, rectangles, fraction) \
    firstprivate(num_props, num_transforms, diagonal)
    for (size_t p_idx = 0; p_idx < num_props; ++p_idx) {
        for (size_t transformIdx = 0; transformIdx < num_transforms; ++transformIdx) {
            for (int colorNuance = colorNuances[0];
                 colorNuance <= colorNuances[1];
                 colorNuance += colorNuances[2]) {

                const float p = prop_values[p_idx];
                const int percentInt =
                    static_cast<int>(std::round(p * 100.0f));

                ImageBuffer modified(
                    baseImage.w,
                    baseImage.h,
                    baseImage.channels
                );
                modified.resetFrom(baseImage);

                partial_transformations_by_proportion_func[transformIdx](
                    modified.get(),
                    p,
                    colorNuance,
                    fraction,
                    rectangles
                );

                std::string outputPath =
                    OutputPathBuilder::buildStandard(
                        partialDirs[transformIdx],
                        baseName,
                        " - ",
                        partialSuffixes[transformIdx],
                        percentInt
                    );

                if (diagonal) {
                    outputPath += " - " +
                        std::to_string(rectangles.size()) +
                        " Squares";
                } else {
                    outputPath += " - Rectangles " +
                        std::to_string(rectangles.front()) +
                        " - " +
                        std::to_string(rectangles.back());
                }

                outputPath += " - CN " +
                    std::to_string(colorNuance) +
                    ".png";

                modified.saveAs(outputPath.c_str());
            }
        }
    }
}


void reverse_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<float>& proportions
) {
    // Precompute proportion values to avoid redundant calculations
    std::vector<float> prop_values;
    prop_values.reserve(
        (proportions[1] - proportions[0]) / proportions[2] + 2
    );

    for (float p = proportions[0]; p <= proportions[1] + 1e-6f; p += proportions[2]) {
        prop_values.push_back(p);
    }

    const int num_threads = computeNumThreads();

    // Parallel processing of proportions for both below and above variants
    #pragma omp parallel for schedule(dynamic) num_threads(num_threads) \
    default(none) \
    shared(prop_values, reversal_step_by_step_output_dirs, reversal_step_by_step_suffixes, baseImage, baseName) \
    firstprivate(num_threads)
    for (size_t p_idx = 0; p_idx < prop_values.size(); ++p_idx) {
        const float p = prop_values[p_idx];
        const int percentInt = static_cast<int>(std::round(p * 100.0f));

        // Skip 50% if already processed (priority case)
        if (std::abs(p - 0.5f) <= 0.001f && p_idx != 0) continue;

        // Create two buffers for concurrent processing
        ImageBuffer modifiedBelow(baseImage.w, baseImage.h, baseImage.channels);
        ImageBuffer modifiedAbove(baseImage.w, baseImage.h, baseImage.channels);

        // Process below variant
        modifiedBelow.resetFrom(baseImage);
        modifiedBelow.get().reverse_by_proportion(p, true);
        std::string outputPathBelow = OutputPathBuilder::buildStandard(
            reversal_step_by_step_output_dirs[0],
            baseName,
            " - ",
            reversal_step_by_step_suffixes[0],
            percentInt
        );
        outputPathBelow += ".png";
        modifiedBelow.saveAs(outputPathBelow.c_str());

        // Process above variant
        modifiedAbove.resetFrom(baseImage);
        modifiedAbove.get().reverse_by_proportion(p, false);
        std::string outputPathAbove = OutputPathBuilder::buildStandard(
            reversal_step_by_step_output_dirs[1],
            baseName,
            " - ",
            reversal_step_by_step_suffixes[1],
            percentInt
        );
        outputPathAbove += ".png";
        modifiedAbove.saveAs(outputPathAbove.c_str());
    }

    // Process 50% case separately if not already handled
    if (std::ranges::find(prop_values, 0.5f) == prop_values.end()) {
        ImageBuffer modifiedBelow(baseImage.w, baseImage.h, baseImage.channels);
        ImageBuffer modifiedAbove(baseImage.w, baseImage.h, baseImage.channels);

        modifiedBelow.resetFrom(baseImage);
        modifiedBelow.get().reverse_by_proportion(0.5f, true);
        std::string outputPathBelow = OutputPathBuilder::buildStandard(
            reversal_step_by_step_output_dirs[0],
            baseName,
            " - ",
            reversal_step_by_step_suffixes[0],
            50
        );
        outputPathBelow += ".png";
        modifiedBelow.saveAs(outputPathBelow.c_str());

        modifiedAbove.resetFrom(baseImage);
        modifiedAbove.get().reverse_by_proportion(0.5f, false);
        std::string outputPathAbove = OutputPathBuilder::buildStandard(
            reversal_step_by_step_output_dirs[1],
            baseName,
            " - ",
            reversal_step_by_step_suffixes[1],
            50
        );
        outputPathAbove += ".png";
        modifiedAbove.saveAs(outputPathAbove.c_str());
    }
}


void edge_detector_image(
	const Image& baseImage,
	const std::string& baseName
) {
	Image img = baseImage;
	img.grayscale_avg();
    const int img_size = img.w*img.h;

    std::vector<uint8_t> grayData(img.w * img.h);

    for (uint64_t k = 0; k < img_size; ++k) {
        grayData[k] = img.data[k * img.channels];  // Extrait le canal de gris
    }

	EdgeDetectorPipeline pipeline(img.w, img.h, 0.09);
	const std::vector<uint8_t>& rgb = pipeline.process(grayData.data());

	Image GT(img.w, img.h, 3);
	std::memcpy(GT.data, rgb.data(), rgb.size());

	const std::string outputPath = std::string(FOLDER_EDGEDETECTOR) + baseName + " GT.png";
	GT.write(outputPath.c_str());
}


void processImageTransforms(
    const std::string& baseName,
    const std::string& inputPath,
    const std::vector<float> &proportions,
    const std::vector<int>& colorNuances,
    const int fraction,
    std::vector<int>& rectanglesToModify,
    const std::vector<int>& tolerance,
    const std::vector<float>& weightOfRGB,
    const bool severalColorsByProportion,
    const bool totalReversal,
    const bool partial,
    const bool partialInDiagonal,
    const bool oneColor,
    const bool average
) {

	if (inputPath.length() >= 4 && [&]() {
		const std::string ext = inputPath.substr(inputPath.length() - 4);
		return ext == ".mp4" || ext == ".MP4";
	}()) {
		std::cout << "MP4 file detected, no transformation applied." << std::endl;
		return;
	}

	std::cout << inputPath << std::endl;

    const Image image(inputPath.c_str());

	edge_detector_image(image, baseName);


    if (oneColor) {
        oneColorTransformations(image, baseName, tolerance, weightOfRGB, average);
    }

	if (severalColorsByProportion) {
		several_colors_transformations_by_proportion(image, baseName,
			proportions, colorNuances);
	}

    if (totalReversal) {
        reverse_transformations_by_proportion(image, baseName, proportions);
    }


    if (partial) {
        partial_transformations_by_proportion(
            image, baseName, proportions, rectanglesToModify, colorNuances, fraction);
    }

    if (partialInDiagonal) {
        std::vector<int> rectanglesInDiagonal = genererRectanglesInDiagonal(fraction);
        partial_transformations_by_proportion(
            image, baseName, proportions, rectanglesInDiagonal, colorNuances, fraction);
    }

}
