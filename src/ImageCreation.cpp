#include "EdgeDetector.h"
#include "ImageCreation.h"

#include <utility>

#include "ColorConfig.h"
#include "VideoProcessing.h"

void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::oneColor) { return; }
    const auto pipeline = OneColorPipeline::forStatic(); // NOLINT(llvmlibc-callee-namespace)

    constexpr int tol_min = parameters::toleranceOneColor[0]; // NOLINT(llvmlibc-callee-namespace)
    constexpr int tol_max = parameters::toleranceOneColor[1]; // NOLINT(llvmlibc-callee-namespace)
    constexpr std::array tol_values = { tol_min, tol_max };

    const size_t total_iterations = pipeline.configCount() * tol_values.size();

#pragma omp parallel default(none) \
    shared(baseImage, pipeline, baseName, total_iterations, tol_values)
    {
        std::string path;
        path.reserve(256);
        std::error_code ec;

#pragma omp for schedule(dynamic)
        for (size_t iter = 0; iter < total_iterations; ++iter) {

            const size_t config_idx = iter / tol_values.size();
            const int tole          = tol_values[iter % tol_values.size()]; // NOLINT(llvmlibc-callee-namespace)

            const auto pipe          = pipeline.buildParams(config_idx);
            const size_t n_outputs = pipeline.outputCount();

            // Check whether all output images for this (config, tolerance) already exist.
            // If so, skip the computation entirely.
            bool all_exist = true;
            for (size_t idx = 0; idx < n_outputs; ++idx) {
                path = OutputPathBuilder::image_one_color(baseName, pipe.weightedColors, tole, idx); // NOLINT(llvmlibc-callee-namespace)
                if (!std::filesystem::exists(path, ec)) { // NOLINT(llvmlibc-callee-namespace)
                    all_exist = false;
                    break;
                }
            }
            if (all_exist) { continue; }

            // At least one output is missing: run the full computation.
            const auto results = pipeline.applyStatic(baseImage, tole, config_idx);

            for (size_t idx = 0; idx < results.size(); ++idx) {
                path = OutputPathBuilder::image_one_color(baseName, pipe.weightedColors, tole, idx); // NOLINT(llvmlibc-callee-namespace)
                // Write only if the file does not already exist.
                if (!std::filesystem::exists(path, ec)) { // NOLINT(llvmlibc-callee-namespace)
                    results[idx].write(path.c_str()); // NOLINT(llvmlibc-callee-namespace)
                }
            }
        }
    }
}


void complete_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::complete_transformation_colors_by_proportion) {return;}

    for (size_t i = 0; i < transformation_params.size(); ++i) {

        const auto& [suffix, output_dir] = total_step_by_step_entries[i]; // NOLINT(llvmlibc-callee-namespace)
        const auto [below, dark] = transformation_params[i]; // NOLINT(llvmlibc-callee-namespace)

        for (int colNua = parameters::colorNuances[0]; // NOLINT(llvmlibc-callee-namespace)
             colNua <= parameters::colorNuances[1]; // NOLINT(llvmlibc-callee-namespace)
             colNua += parameters::colorNuances[2]) { // NOLINT(llvmlibc-callee-namespace)
            constexpr float proportion = 0.5F;

            ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
            modified.resetFrom(baseImage);

            modified.get().proportion_complete(proportion, colNua, dark, below);

            const std::string path =
                OutputPathBuilder::image_complete( // NOLINT(llvmlibc-callee-namespace)
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
        const auto [below, dark] = transformation_params[transformIdx];
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
    const bool diagonal = (!rectangles.empty() && rectangles[0] == -1);;

    // Capture rectangles, fraction, diagonal by value — safe across threads
    auto apply = [rectangles](Image& img, const float proportion, const size_t transformIdx, const int
        colNua) -> bool {
        if (proportion <= 0.0F) { return false;
}
        const auto [below, dark] = transformation_params[transformIdx];
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
    if constexpr (!parameters::totalReversal) {return;}
    const std::vector entries(
        reversal_step_by_step_entries.begin(),
        reversal_step_by_step_entries.end()
    );

    // Skip duplicate p=0.5: the reversal is symmetric, processing it twice is wasteful
    auto applyWithSkip = [](Image& img, const float proportion,
                        const size_t i, const int /*colNua*/) -> bool {
        if (proportion <= 0.0F) { return false; }
        img.reverse_by_proportion(proportion, reversal_below_flag(i)); // NOLINT(llvmlibc-callee-namespace)
        return true;
    };

    auto buildPath = [](const std::string& dir, const std::string& base,
                        const std::string& suffix, const float proportion, int /*cn*/) {
        return OutputPathBuilder::image_reverse(dir, base, suffix, proportion);
    };

    run_transformations_by_proportion(
        baseImage, baseName, entries,
        applyWithSkip, buildPath,
        {}
        // no colorNuances → nullptr by default
    );

}

void edge_detector_image(
	const Image& baseImage,
	const std::string& baseName
) {
	Image img = baseImage;
	img.grayscale_avg();
    const int img_size = img.w*img.h;

    std::vector<uint8_t> grayData(img.w * img.h);

    for (uint64_t k = 0; std::cmp_less(k , img_size); ++k) {
        grayData[k] = img.data[k * img.channels];  // Extrait le canal de gris
    }

	EdgeDetectorPipeline pipeline(img.w, img.h, 0.09);
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
#include "EdgeDetector.h"
#include "ImageCreation.h"

#include "ColorConfig.h"

void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::oneColor) { return; }
    const OneColorPipeline pipeline(true);

    constexpr int tol_min = parameters::toleranceOneColor[0];
    constexpr int tol_max = parameters::toleranceOneColor[1];

    constexpr std::array tol_values = { tol_min, tol_max };
    constexpr int num_tole = tol_values.size();

    const size_t total_iterations = pipeline.configCount() * num_tole;

#pragma omp parallel default(none) \
    shared(baseImage, pipeline, baseName, total_iterations, tol_values, num_tole)
    {
        std::string path;
        path.reserve(256);

#pragma omp for schedule(dynamic)
        for (size_t iter = 0; iter < total_iterations; ++iter) {

            const size_t config_idx = iter / num_tole;
            const int tole = tol_values[iter % num_tole];

            const auto p = pipeline.buildParams(config_idx);
            const auto results = pipeline.apply(baseImage, tole, config_idx);

            for (size_t idx = 0; idx < results.size(); ++idx) {
                path = OutputPathBuilder::image_one_color(baseName, p.weightedColors, tole, idx);
                results[idx].write(path.c_str());
            }
        }
    }
}


void complete_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName
) {
    if constexpr (!parameters::complete_transformation_colors_by_proportion) {return;}

    for (size_t i = 0; i < transformation_params.size(); ++i) {

        const auto& [suffix, output_dir] = total_step_by_step_entries[i];
        const auto [below, dark] = transformation_params[i];

        for (int cn = parameters::colorNuances[0];
             cn <= parameters::colorNuances[1];
             cn += parameters::colorNuances[2]) {
            constexpr float p = 0.5f;

            ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
            modified.resetFrom(baseImage);

            modified.get().proportion_complete(p, cn, dark, below);

            const std::string path =
                OutputPathBuilder::image_complete(
                    folder_50,
                    baseName,
                    suffix,
                    p,
                    cn
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
    auto apply = [](Image& img, const float p, const size_t transformIdx, const int cn) -> bool {
        if (p <= 0.0f || p >= 1.0f) return false;
        const auto [below, dark] = transformation_params[transformIdx];
        img.proportion_complete(p, cn, dark, below);
        return true;
    };

    // buildPath: forward to the OutputPathBuilder, ignore cn=0 sentinel when no nuance
    auto buildPath = [](const std::string& dir, const std::string& base,
                        const std::string& suffix, const float p, const int cn) {
        return OutputPathBuilder::image_complete(dir, base, suffix, p, cn);
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
    const bool diagonal = (!rectangles.empty() && rectangles[0] == -1);;

    // Capture rectangles, fraction, diagonal by value — safe across threads
    auto apply = [rectangles](Image& img, const float p, const size_t transformIdx, const int cn) -> bool {
        if (p <= 0.0f) return false;
        const auto [below, dark] = transformation_params[transformIdx];
        img.proportion_region_fraction(p, cn, parameters::fraction, rectangles, dark, below);
        return true;
    };

    auto buildPath = [diagonal, rectangles](const std::string& dir, const std::string& base,
                                             const std::string& suffix, const float p, const int cn) {
        return OutputPathBuilder::image_partial(dir, base, suffix, p, cn, diagonal, rectangles);
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
    if constexpr (!parameters::totalReversal) {return;}
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
        return OutputPathBuilder::image_reverse(dir, base, suffix, p);
    };

    run_transformations_by_proportion(
        baseImage, baseName, entries,
        applyWithSkip, buildPath
        // no colorNuances → nullptr by default
    );

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

bool processImageTransforms(
    const std::string& baseName,
    const std::string& inputPath
) {

    if (is_mp4_file(inputPath)) {
        Logger::log("MP4 file detected, no transformation applied.");
        return false;
    }

    Logger::log(inputPath);

    const Image image(inputPath.c_str());

    edge_detector_image(image, baseName);
    oneColorTransformations(image, baseName);
    complete_transformations_by_proportion(image, baseName);
    reverse_transformations_by_proportion(image, baseName);


    for (const bool useDiagonal : {false, true}) {
        if (useDiagonal ? !parameters::partialInDiagonal : !parameters::partial)
            continue;

        const std::vector<int> rectangles = useDiagonal
            ? generateDiagonalRectangles()
            : decode_rectangles().second;

        partial_transformations_by_proportion(image, baseName, rectangles);
    }
    return true;

}
