#include "ImageProcessing.h"
#include "TransformationsConfig.h"
#include <vector>
#include <functional>
#include <string>
#include <cstring>
#include <immintrin.h> // For AVX2 intrinsics
#include <format>

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

std::string makeWeightedColors(float r, float g, float b) {
	std::string s;
	s.reserve(32); // pour éviter plusieurs allocations

	s += ' ';
	s += std::format("{:.2f}", r);
	s += '-';
	s += std::format("{:.2f}", g);
	s += '-';
	s += std::format("{:.2f}", b);

	return s;
}

void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& tolerance
) {

	const std::string ONE_COLOR_FOLDER = std::string(OUTPUT_FOLDER) + "One Color/";
	const std::string DIFFMAP_FOLDER = std::string(OUTPUT_FOLDER) + "Diffmap/";

	ImageBuffer buffer(baseImage.w, baseImage.h, baseImage.channels);

	constexpr float weightOfRed = 0.25f;
	constexpr float weightOfGreen = 1.0f;
	constexpr float weightOfBlue = 0.75f;
	const auto [r_third, g_third, b_third,
		r_half, g_half, b_half,
		r_full, g_full, b_full] =
		calculateWeightedColors(weightOfRed, weightOfGreen, weightOfBlue);

	const std::string prefix0 = ONE_COLOR_FOLDER + baseName + " - Average 0 - Tolerance ";
	const std::string prefix1 = ONE_COLOR_FOLDER + baseName + " - Average 1 - Tolerance ";
	const std::string weightedColors = makeWeightedColors(weightOfRed, weightOfGreen, weightOfBlue);
	const std::string suffix = ".png";

    for (int tole = tolerance[0]; tole <= tolerance[1]; tole += tolerance[2]) {
        buffer.resetFrom(baseImage);

        buffer.get().simplify_to_dominant_color_combinations_without_average(
            tole, r_third, g_third, b_third,
            r_half, g_half, b_half,
            r_full, g_full, b_full
        );

        std::string path;
        path.reserve(256);
        path = prefix0;
        path += std::to_string(tole);
        path += weightedColors;
        path += suffix;

        buffer.saveAs(path.c_str());

        buffer.get().simplify_to_dominant_color_combinations_with_average(tole);

        path.clear();
        path = prefix1;
        path += std::to_string(tole);
        path += suffix;

        buffer.saveAs(path.c_str());
    }

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

void several_colors_transformations_AVX2(
	const Image& baseImage,
	const std::string& baseName,
	const int firstThreshold,
	const int lastThreshold,
	const int step,
	const int firstColorNuance,
	const int lastColorNuance,
	const int stepColorNuance
) {
	ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
	std::ostringstream oss;

	for (int currentThreshold = firstThreshold; currentThreshold <= lastThreshold; currentThreshold += step) {
		const int threshold_div3 = currentThreshold / 3;

		for (size_t transformIdx = 0; transformIdx < colors_nuances_transformations_AVX2.size(); ++transformIdx) {
			const auto& transform = colors_nuances_transformations_AVX2[transformIdx];
			const auto& outputDir = total_step_by_step_output_dirs[transformIdx];
			const auto& suffix = total_step_by_step_suffixes[transformIdx];

			for (int currentColorNuance = firstColorNuance;
				currentColorNuance <= lastColorNuance;
				currentColorNuance += stepColorNuance) {

				modified.resetFrom(baseImage);
				transform(modified.get(), currentThreshold, static_cast<std::uint8_t>(currentColorNuance));

				// Construction du chemin
				oss.str("");
				oss << outputDir << baseName << " - " << suffix << " - "
					<< threshold_div3 << " - CN " << currentColorNuance << ".png";

				modified.saveAs(oss.str().c_str());
				}
		}
	}
}

void several_colors_transformations_by_threshold(
    const Image& baseImage,
    const std::string& baseName,
    const int firstThreshold,
    const int lastThreshold,
    const int step,
    const std::vector<int>& colorNuances
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    auto process = [&](const int currentThreshold, const bool duplicate120) {
        for (size_t transformIdx = 0; transformIdx < transformations_by_threshold.size(); ++transformIdx) {
            for (int currentColorNuance = colorNuances[0];
                 currentColorNuance <= colorNuances[1];
                 currentColorNuance += colorNuances[2]) {

                modified.resetFrom(baseImage);

                transformations_by_threshold[transformIdx](
                    modified.get(), currentThreshold, currentColorNuance
                );

                // Sortie standard
                std::string outputPath = OutputPathBuilder::buildStandard(
                    total_step_by_step_output_dirs[transformIdx],
                    baseName,
                    " - ",
                    total_step_by_step_suffixes[transformIdx],
                    currentThreshold / 3
                );
                outputPath += " - CN " + std::to_string(currentColorNuance) + ".png";
                modified.saveAs(outputPath.c_str());

                // Duplication spécifique pour 120
                if (duplicate120) {
                    std::string path120 = OutputPathBuilder::build120(
                        FOLDER_120,
                        baseName,
                        " - ",
                        total_step_by_step_suffixes[transformIdx]
                    );
                    path120 += "120 - CN " + std::to_string(currentColorNuance) + ".png";
                    modified.saveAs(path120.c_str());
                }
            }
        }
    };

    process(360, true);

    for (int currentThreshold = firstThreshold;
         currentThreshold <= lastThreshold;
         currentThreshold += step) {

        if (currentThreshold == 360) {
            continue;
        }

        process(currentThreshold, false);
    }
}

void several_colors_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<float>& proportions,
    const std::vector<int>& colorNuances
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    auto process = [&](const float currentProportion, const bool duplicate50) {
        for (size_t transformIdx = 0; transformIdx < transformations_by_proportion.size(); ++transformIdx) {
            for (int currentColorNuance = colorNuances[0];
                 currentColorNuance <= colorNuances[1];
                 currentColorNuance += colorNuances[2]) {

                modified.resetFrom(baseImage);

                transformations_by_proportion[transformIdx](
                    modified.get(), currentProportion, currentColorNuance
                );

                // Sortie standard
                std::string outputPath = OutputPathBuilder::buildStandard(
                    total_step_by_step_output_dirs[transformIdx],
                    baseName,
                    " - ",
                    total_step_by_step_suffixes[transformIdx],
                    static_cast<int>(currentProportion * 100)
                );
                outputPath += "% - CN " + std::to_string(currentColorNuance) + ".png";
                modified.saveAs(outputPath.c_str());

                // Duplication spécifique pour 0.5
                if (duplicate50) {
                    std::string path50 = OutputPathBuilder::build120(
                        FOLDER_120,
                        baseName,
                        " - ",
                        total_step_by_step_suffixes[transformIdx]
                    );
                    path50 += " 50% - CN " + std::to_string(currentColorNuance) + ".png";
                    modified.saveAs(path50.c_str());
                }
            }
        }
    };

    process(0.5f, true);

	const int numSteps = static_cast<int>((proportions[1] - proportions[0]) / proportions[2]) + 1;
	for (int i = 0; i < numSteps; ++i) {
		const float currentProportion = proportions[0] + i * proportions[2];
		if (std::abs(currentProportion - 0.5f) < 0.001f) {
			continue;
		}
		process(currentProportion, false);
	}

}


void several_colors_partial_transformations(
    const Image& baseImage,
    const std::vector<GenericTransformationFuncWithColorNuances>& transforms,
    const std::string& baseName,
    const int firstThreshold,
    const int lastThreshold,
    const int step,
    const int fraction,
    std::vector<int>& rectangles,
    const std::vector<int>& colorNuances
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    const std::vector<std::string> partialDirs = generatePartialOutputDirs();
    const std::vector<std::string> partialSuffixes = generatePartialSuffixes();

    bool diagonal = false;

    if (rectangles[0] == -1) {
        rectangles.erase(rectangles.begin());
        diagonal = true;
    }

    for (int currentThreshold = firstThreshold; currentThreshold <= lastThreshold; currentThreshold += step) {
        for (size_t transformIdx = 0; transformIdx < partialTransformationsFunc.size(); ++transformIdx) {
            for (int currentColorNuance = colorNuances[0];
                currentColorNuance <= colorNuances[1]; currentColorNuance += colorNuances[2]) {

                modified.resetFrom(baseImage);

                transforms[transformIdx](modified.get(), currentThreshold,
                    currentColorNuance, fraction, rectangles);

                std::string outputPath = OutputPathBuilder::buildStandard(
                    partialDirs[transformIdx],
                    baseName,
                    " - ",
                    partialSuffixes[transformIdx],
                    currentThreshold / 3
                );

                if (diagonal) {
                    outputPath += " - " + std::to_string(rectangles.size()) + " Squares ";
                } else {
                    outputPath += " - Rectangles " + std::to_string(rectangles[0]) +
                    " - " + std::to_string(rectangles.back());
                }
                outputPath += " CN " +
                        std::to_string(currentColorNuance) + ".png";
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

	EdgeDetectorPipeline pipeline(img.w, img.h, 0.09);
	const std::vector<uint8_t>& rgb = pipeline.process(img.data);

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
    const bool severalColorsByThreshold,
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
        oneColorTransformations(image, baseName, tolerance);
    }

    if (severalColorsByThreshold) {
        several_colors_transformations_by_threshold(image, baseName, first_threshold, last_threshold, step,
            colorNuances);
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
        std::vector altParams = {first_threshold, last_threshold, step};
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
        several_colors_partial_transformations(
            image,
            wrapPartialTransformsWithRectangles(partialTransformationsFunc),
            baseName,
            first_threshold,
            last_threshold,
            step,
            fraction,
            rectanglesToModify,
            colorNuances
        );
    }

    if (partialInDiagonal) {
        std::vector<int> rectanglesInDiagonal = genererRectanglesInDiagonal(fraction);
        several_colors_partial_transformations(
            image,
            wrapPartialTransformsWithRectangles(partialTransformationsFunc),
            baseName,
            first_threshold,
            last_threshold,
            step,
            fraction,
            rectanglesInDiagonal,
            colorNuances
        );
    }

}

