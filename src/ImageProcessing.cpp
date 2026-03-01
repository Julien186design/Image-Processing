#include "Image.h"
#include "ImageProcessing.h"
#include "ProcessingConfig.h"
#include <vector>
#include <string>
#include <cstring>
#include <immintrin.h>
#include <format>
#include <execution>

void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& tolerance,
    const std::vector<float>& weightOfRGB,
    const bool average
) {

    const OneColorPipeline pipeline(weightOfRGB, average, true, true);

    const int num_tole        = (tolerance[1] - tolerance[0]) / tolerance[2] + 1;
    const size_t total_iterations = pipeline.configCount() * num_tole;

    #pragma omp parallel default(none) \
    shared(baseImage, pipeline, tolerance, baseName, num_tole, total_iterations)
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
            if (pipeline.average) {
                OneColorPipeline::applyWithAverage(buffer.get(), tole, pipeline.configs[config_idx]);
            } else {
                OneColorPipeline::applyWithoutAverage(buffer.get(), tole, p);
            }
            path = OutputPathBuilder::image_one_color(pipeline.average, baseName, p.weightedColors, tole);
            buffer.saveAs(path.c_str());
        }
    }
}


void complete_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const PropRange& proportions,
    const std::vector<int>& colorNuances
) {
    // Build entries from the global step-by-step table
    const std::vector entries(
        total_step_by_step_entries.begin(),
        total_step_by_step_entries.end()
    );

    // apply: delegate to proportion_complete using per-transform params
    auto apply = [](Image& img, const float p, const size_t transformIdx, const int cn) -> bool {
        if (p <= 0.0f || p >= 1.0f) return false;
        const auto [below, dark] = transformation_params[transformIdx];
        img.proportion_complete(p, cn, dark, below);
        return true;
    };

    // buildPath: forward to the OutputPathBuilder, ignore cn=0 sentinel when no nuance
    auto buildPath = [](const std::string& dir, const std::string& base,
                        const std::string& suffix, const float p, const int cn) {
        return OutputPathBuilder::complete(dir, base, suffix, p, cn);
    };

    run_transformations_by_proportion(
        baseImage, baseName, proportions, entries,
        apply, buildPath,
        &colorNuances  // pass nuances → activates inner color loop
    );
}

void partial_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const PropRange& proportions,
    const std::vector<int>& first_and_last_rectangles,
    const std::vector<int>& colorNuances,
    const int fraction
) {
    const std::vector<TransformationEntry> partialEntries =
        generatePartialEntries(total_step_by_step_entries);

    const auto [diagonal, rectangles] = decode_rectangles(first_and_last_rectangles);

    // Capture rectangles, fraction, diagonal by value — safe across threads
    auto apply = [rectangles, fraction](Image& img, const float p, const size_t transformIdx, const int cn) -> bool {
        if (p <= 0.0f) return false;
        const auto [below, dark] = transformation_params[transformIdx];
        img.proportion_region_fraction(p, cn, fraction, rectangles, dark, below);
        return true;
    };

    auto buildPath = [diagonal, rectangles](const std::string& dir, const std::string& base,
                                             const std::string& suffix, const float p, const int cn) {
        return OutputPathBuilder::partial(dir, base, suffix, p, cn, diagonal, rectangles);
    };

    run_transformations_by_proportion(
        baseImage, baseName, proportions, partialEntries,
        apply, buildPath,
        &colorNuances
    );
}

void reverse_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const PropRange& proportions
) {
    const std::vector entries(
        reversal_step_by_step_entries.begin(),
        reversal_step_by_step_entries.end()
    );

    // Skip duplicate p=0.5: the reversal is symmetric, processing it twice is wasteful
    auto applyWithSkip = [](Image& img, const float p, const size_t i, const int cn) -> bool {
        if (p <= 0.0f) return false;
        const bool below = (i == 0);
        img.reverse_by_proportion(p, below);
        return true;
    };

    auto buildPath = [](const std::string& dir, const std::string& base,
                        const std::string& suffix, const float p, int /*cn*/) {
        return OutputPathBuilder::reverse(dir, base, suffix, p);
    };


    run_transformations_by_proportion(
        baseImage, baseName, proportions, entries,
        applyWithSkip, buildPath
        // no colorNuances → nullptr by default
    );

    // Handle p=0.5 separately if it falls outside the proportion range
    const size_t num_props =
        static_cast<size_t>((proportions.stop - proportions.start) / proportions.step) + 1;

    if (const float last_p = proportions.start + (num_props - 1) * proportions.step; last_p < 0.5f || proportions.start > 0.5f) {
        reverse_at_proportion(baseImage, baseName, 0.5f);
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

	const std::string outputPath = OutputPathBuilder::image_edge_detector(baseName);
	GT.write(outputPath.c_str());
}


void processImageTransforms(
    const std::string& baseName,
    const std::string& inputPath,
    const PropRange& proportions,
    const std::vector<int>& colorNuances,
    const std::vector<int>& rectangles,
    const std::vector<int>& tolerance,
    const std::vector<float>& weightOfRGB
) {

    if (is_mp4_file(inputPath)) {
        std::cout << "MP4 file detected, no transformation applied." << std::endl;
        return;
    }

	std::cout << inputPath << std::endl;

    const Image image(inputPath.c_str());

	edge_detector_image(image, baseName);

    if (parameters::oneColor) {
        oneColorTransformations(image, baseName, tolerance, weightOfRGB, parameters::average);
    }

	if (parameters::complete_transformation_colors_by_proportion) {
		complete_transformations_by_proportion(image, baseName, proportions, colorNuances);
	}

    if (parameters::totalReversal) {
        reverse_transformations_by_proportion(image, baseName, proportions);
    }

    if (parameters::partial) {
        partial_transformations_by_proportion(
            image, baseName, proportions, rectangles, colorNuances, parameters::fraction);
    }

    if (parameters::partialInDiagonal) {
        const std::vector<int> rectanglesInDiagonal = genererRectanglesInDiagonal(parameters::fraction);
        partial_transformations_by_proportion(
            image, baseName, proportions, rectanglesInDiagonal, colorNuances, parameters::fraction);
    }

}
