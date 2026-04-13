#include "EdgeDetector.h"
#include "ImageCreation.h"
#include "ColorConfig.h"

#include <utility>

void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::oneColor) { return; }
    const auto pipeline = OneColorPipeline::forStatic();

    constexpr int tol_min = std::get<0>(parameters::toleranceOneColor);
    constexpr int tol_max = std::get<1>(parameters::toleranceOneColor);
    constexpr std::array tol_values = { tol_min, tol_max };

    const size_t total_iterations = pipeline.configCount() * tol_values.size();
    std::cout << "pipe " << pipeline.configCount() << " configs" << std::endl;
    std::cout << "tol_v " << tol_values.size() << std::endl;
    std::cout << "total " << total_iterations << std::endl;

#pragma omp parallel default(none) \
    shared(baseImage, pipeline, baseName, total_iterations, tol_values)
    {
        std::string path;
        path.reserve(256);
        std::error_code ec;

#pragma omp for schedule(dynamic)
        for (size_t iter = 0; iter < 14; ++iter) {

            const size_t config_idx = iter / tol_values.size();
            const int tole          = tol_values.at(iter % tol_values.size());

            const auto [weightedColors]          = pipeline.buildParams(config_idx);

            // Check whether all output images for this (config, tolerance) already exist.
            // If so, skip the computation entirely.
            constexpr size_t expected = 5;
            bool all_exist = true;
            for (size_t idx = 0; idx < expected; ++idx) {
                path = OutputPathBuilder::image_one_color(baseName, weightedColors, tole, idx);
                if (!std::filesystem::exists(path, ec)) {
                    all_exist = false;
                    break;
                }
            }
            if (all_exist) { continue; }

            size_t idx = 0;
            baseImage.simplify_to_dominant_color_combinations(
                tole, &pipeline.configs.at(config_idx), {},
                [&](Image&& result) {
                    path = OutputPathBuilder::image_one_color(
                        baseName, weightedColors, tole, idx);
                    if (!std::filesystem::exists(path, ec)) {
                        result.write(path.c_str());
                    }
                    return ++idx < expected;
                }
            );
        }
    }
}


void complete_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::complete_transformation_colors_by_proportion) {return;}

    for (size_t i = 0; i < transformation_params.size(); ++i) {

        const auto& [suffix, output_dir] = total_step_by_step_entries.at(i);
        const auto [below, dark] = transformation_params.at(i);

        for (int colNua = parameters::colorNuances.at(0);
             colNua <= parameters::colorNuances.at(1);
             colNua += parameters::colorNuances.at(2)) {
            constexpr float proportion = 0.5F;

            ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
            modified.resetFrom(baseImage);

            modified.get().proportion_complete(proportion, colNua, dark, below);

            const std::string path =
                OutputPathBuilder::image_complete(
                    folder_50,
                    baseName,
                    suffix,
                    proportion,
                    colNua
                );

            modified.saveAs(path.c_str());
             }
    }

    // Build entries from the global step-by-step table
    const std::vector entries(
        total_step_by_step_entries.begin(),
        total_step_by_step_entries.end()
    );

    // apply: delegate to proportion_complete using per-transform params
    auto apply = [](Image& img, const float proportion, const size_t transformIdx, const int colNua) -> bool {
        if (proportion <= 0.0F || proportion >= 1.0F) { return false;
}
        const auto [below, dark] = transformation_params.at(transformIdx);
        img.proportion_complete(proportion, colNua, dark, below);
        return true;
    };

    // buildPath: forward to the OutputPathBuilder, ignore colNua=0 sentinel when no nuance
    auto buildPath = [](const std::string& dir, const std::string& base,
                        const std::string& suffix, const float proportion, const int colNua) {
        return OutputPathBuilder::image_complete(dir, base, suffix, proportion, colNua);
    };

    run_transformations_by_proportion(
        baseImage, baseName, entries,
        apply, buildPath,
        &parameters::colorNuances  // pass nuances → activates inner color loop
    );
}

void partial_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int> &rectangles
) {
    const std::vector<TransformationEntry> partialEntries =
        generatePartialEntries(total_step_by_step_entries);

    // Diagonal mode is indicated by the -1 sentinel at index 0
    const bool diagonal = (!rectangles.empty() && rectangles.at(0) == -1);;

    // Capture rectangles, fraction, diagonal by value — safe across threads
    auto apply = [rectangles](Image& img, const float proportion, const size_t transformIdx, const int
        colNua) -> bool {
        if (proportion <= 0.0F) { return false;
}
        const auto [below, dark] = transformation_params.at(transformIdx);
        img.proportion_region_fraction(proportion, colNua, parameters::fraction, rectangles, dark, below);
        return true;
    };

    auto buildPath = [diagonal, rectangles](const std::string& dir, const std::string& base,
                                             const std::string& suffix, const float proportion, const int colNua) {
        return OutputPathBuilder::image_partial(dir, base, suffix, proportion, colNua, diagonal, rectangles);
    };

    run_transformations_by_proportion(
        baseImage, baseName, partialEntries,
        apply, buildPath,
        &parameters::colorNuances
    );
}

void reverse_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::totalReversal) { return; }

    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
    modified.resetFrom(baseImage);
    modified.get().reverse_by_proportion(1.0F, true);
    modified.saveAs(
        OutputPathBuilder::image_reverse(baseName, 1.0F).c_str()
    );
}

void edge_detector_image(
	const Image& baseImage,
	const std::string& baseName
) {
	Image img = baseImage;
	img.grayscale_avg();
    const size_t img_size = static_cast<size_t>(img.w) * img.h;

    std::vector<uint8_t> grayData(img_size);

    const std::span<const uint8_t> dataSpan(img.data, img_size * img.channels);

    for (size_t k = 0, src = 0; k < img_size; ++k, src += static_cast<size_t>(img.channels)) {
        grayData[k] = dataSpan[src];
    }

	EdgeDetectorPipeline pipeline(img.w, img.h);
	const std::vector<uint8_t>& rgb = pipeline.process(grayData.data());

	Image GT(img.w, img.h, 3);
	std::memcpy(GT.data, rgb.data(), rgb.size());

	const std::string outputPath = OutputPathBuilder::image_edge_detector(baseName);
	GT.write(outputPath.c_str());
}

bool processImageTransforms(
    const std::string& baseName,
    const std::string& inputPath
) {

    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected, no transformation applied.");
        return false;
    }

    Logger::log(inputPath);

    const Image image(inputPath.c_str(), 0);

    edge_detector_image(image, baseName);
    oneColorTransformations(image, baseName);
    complete_transformations_by_proportion(image, baseName);
    reverse_transformations_by_proportion(image, baseName);


    for (const bool useDiagonal : {false, true}) {
        if (useDiagonal ? !parameters::partialInDiagonal : !parameters::partial) {
            continue;
}

        const std::vector<int> rectangles = useDiagonal
            ? generateDiagonalRectangles()
            : decode_rectangles().second;

        partial_transformations_by_proportion(image, baseName, rectangles);
    }
    return true;

}
