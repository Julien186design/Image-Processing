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
#include <omp.h>

using GenericTransformationFunc = std::function<void(Image&, int, const std::vector<int>&)>;

struct TransformationContext {
    const Image& baseImage;
    const std::string& baseName;
    int firstThreshold;
    int lastThreshold;
    int step;
    bool saveAt120;
    std::string folder120;
};


void applyAndSaveGenericTransformations(
    const Image& baseImage,
    const std::vector<GenericTransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const std::string& transformationType,
    const int threshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120,
    const std::vector<int>& additionalParams
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int currentThreshold = threshold; currentThreshold <= lastThreshold; currentThreshold += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {
            // LOGGING
            // printf("  Transformation %zu/%zu: %s\n", i+1, transforms.size(), suffixes[i].c_str());

            modified.resetFrom(baseImage);

            transforms[i](modified.get(), currentThreshold, additionalParams);

            std::string outputPath = OutputPathBuilder::buildStandard(
                outputDirs[i],
                baseName,
                transformationType,
                suffixes[i],
                currentThreshold / 3
            );
            modified.saveAs(outputPath.c_str());

            if (saveAt120 && currentThreshold == 360) {
                std::string specialPath = OutputPathBuilder::build120(
                    folder120,
                    baseName,
                    transformationType,
                    suffixes[i]
                );
                modified.saveAs(specialPath.c_str());
            }
        }

    }
}


void applyTransformationsWrapper(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const int threshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120
) {
    applyAndSaveGenericTransformations(
        baseImage,
        wrapSimpleTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " - ",
        threshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        {}
    );
}



void applyAndSaveReversalTransformations(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const int threshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120
) {
    applyAndSaveGenericTransformations(
        baseImage,
        wrapSimpleTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " Reversal - ",
        threshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        {}
    );
}

void applyAndSaveAlternatingTransformations(
    const Image& baseImage,
    const std::vector<AlternatingTransformation>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const int firstThreshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120
) {
    const std::vector<int> additionalParams = {firstThreshold, lastThreshold, step};

    applyAndSaveGenericTransformations(
        baseImage,
        wrapAlternatingTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " Alternating - ",
        firstThreshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        additionalParams
    );
}


void removeColors(
	const Image& baseImage,
	const std::string& baseName
) {
	ImageBuffer buffer(baseImage.w, baseImage.h, baseImage.channels);
	buffer.resetFrom(baseImage);
	buffer.get().color_mask(1.0f, 0.0f, 1.0f);

	const std::string outputPath = std::string(OUTPUT_FOLDER) + baseName + ".png";
	buffer.saveAs(outputPath.c_str());
}



void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& tolerance,
    const std::vector<float>& weightOfRGB
) {
    const std::string ONE_COLOR_FOLDER = std::string(OUTPUT_FOLDER) + "One Color/";
    const std::string prefix0 = ONE_COLOR_FOLDER + baseName + " - Average 0 - Tolerance ";
    const std::string prefix1 = ONE_COLOR_FOLDER + baseName + " - Average 1 - Tolerance ";
    const std::string suffix = ".png";

    const std::vector<std::vector<float>> configs = generateColorConfigs(weightOfRGB[3]);

    for (const auto& config : configs) {
        if ((config[0] == 0.0f && config[1] == 1.0f) ||
            (config[0] == 1.0f && config[1] == 0.0f)) {
            std::cout << "[";
            for (size_t i = 0; i < config.size(); ++i) {
                std::cout << "  " << config[i];
                if (i != config.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
    }

    const int num_tole = (tolerance[1] - tolerance[0]) / tolerance[2] + 1;
    const size_t total_iterations = configs.size() * num_tole;

    #pragma omp parallel
    {
        ImageBuffer buffer(baseImage.w, baseImage.h, baseImage.channels);
        std::string path;
        path.reserve(256);

        #pragma omp for schedule(dynamic)
        for (size_t iter = 0; iter < total_iterations; ++iter) {
            const size_t config_idx = iter / num_tole;
            const int tole = tolerance[0] + (iter % num_tole) * tolerance[2];

            const auto& config = configs[config_idx];
            const auto [r_third, g_third, b_third,
                r_half, g_half, b_half,
                r_full, g_full, b_full] =
                calculateWeightedColors(config);

            const std::string weightedColors = makeWeightedColors(config);

            buffer.resetFrom(baseImage);

            buffer.get().simplify_to_dominant_color_combinations_without_average(
                tole, r_third, g_third, b_third,
                r_half, g_half, b_half,
                r_full, g_full, b_full
            );

            path.clear();
            path = prefix0;
            path += std::to_string(tole);
            path += weightedColors;
            path += suffix;

            buffer.saveAs(path.c_str());

            buffer.get().simplify_to_dominant_color_combinations_with_average(tole);

            path.clear();
            path = prefix1;
            path += std::to_string(tole);
            path += weightedColors;
            path += suffix;

            buffer.saveAs(path.c_str());
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

    // Compute number of threads locally (no global side effect)
    const int max_threads = omp_get_max_threads();
    const int num_threads = std::max(1, max_threads - THREAD_OFFSET);

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

    #pragma omp parallel for collapse(3) schedule(dynamic) num_threads(num_threads)
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
    const std::vector<int>& thresholdsAndStep,
    const std::vector<float> &proportions,
    const std::vector<int>& colorNuances,
    const int fraction,
    std::vector<int>& rectanglesToModify,
    const std::vector<int>& tolerance,
    const std::vector<float> &weightOfRGB,
    const bool severalColorsByProportion,
    const bool totalBlackAndWhite,
    const bool totalReversal,
    const bool partial,
    const bool partialInDiagonal,
    const bool alternatingBlackAndWhite,
    const bool oneColor
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

    const int first_threshold = 3 * thresholdsAndStep[0];
    const int last_threshold = 3 * thresholdsAndStep[1];
    const int step = 3 * thresholdsAndStep[2];


	/*
	removeColors(image, baseName);
	*/

	edge_detector_image(image, baseName);


    if (oneColor) {
        oneColorTransformations(image, baseName, tolerance, weightOfRGB);
    }


	if (severalColorsByProportion) {
		several_colors_transformations_by_proportion(image, baseName,
			proportions, colorNuances);
	}

    if (totalBlackAndWhite) {
        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_black_and_white_transformations),
            total_black_and_white_output_dirs,
            total_black_and_white_suffixes,
            baseName,
            " - ",
            first_threshold,
            last_threshold,
            step,
            true,
            FOLDER_120,
            {}
        );
    }

    if (totalReversal) {
        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_reversal_step_by_step_transformations),
            reversal_step_by_step_output_dirs,
            reversal_step_by_step_suffixes,
            baseName,
            " - ",
            first_threshold,
            last_threshold,
            step,
            true,
            FOLDER_120,
            {}
        );
    }

    if (alternatingBlackAndWhite) {
        const std::vector altParams = {first_threshold, last_threshold, step};
        applyAndSaveGenericTransformations(
            image,
            wrapAlternatingTransforms(total_alternating_black_and_white_transformations),
            total_step_by_step_output_dirs,
            total_step_by_step_suffixes,
            baseName,
            " Alternating - ",
            first_threshold,
            last_threshold,
            step,
            true,
            FOLDER_120,
            altParams
        );
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

