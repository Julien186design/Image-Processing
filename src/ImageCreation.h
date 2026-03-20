#ifndef IMAGE_PROCESSING_IMAGECREATION_H
#define IMAGE_PROCESSING_IMAGECREATION_H

#include "ImageBuffer.h"

#include <string>

template<typename ApplyFunc, typename PathFunc>
void run_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<TransformationEntry>& entries,
    ApplyFunc apply,
    PathFunc buildPath,
    const std::array<int,3>* colorNuances = nullptr
) {
    const int num_threads = computeNumThreads();
    // Executes one (proportion, entry, color nuance) triplet:
    // resets buffer, applies transform, saves if apply() did not skip.
    auto process_one = [&](const float p, const size_t i, const int cn) {
        const auto& [suffix, output_dir] = entries[i];
        ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
        modified.resetFrom(baseImage);
        if (!apply(modified.get(), p, i, cn)) return;
        modified.saveAs(buildPath(output_dir, baseName, suffix, p, cn).c_str());
    };

#pragma omp parallel for schedule(dynamic) num_threads(num_threads) \
default(none) \
shared(baseImage, baseName, entries, colorNuances, apply, buildPath, process_one)
    for (size_t p_idx = 0; p_idx < parameters::numProportionSteps; ++p_idx) {
        const float p = parameters::proportions[0] + static_cast<float>(p_idx) * parameters::proportions[2];
        for (size_t i = 0; i < entries.size(); ++i) {
            if (colorNuances) {
                for (int cn = (*colorNuances)[0];
                     cn <= (*colorNuances)[1];
                     cn += (*colorNuances)[2])
                    process_one(p, i, cn);
            } else {
                process_one(p, i, 0);
            }
        }
    }
}

bool processImageTransforms(
    const std::string& baseName ,
    const std::string& inputPath
);

#endif //IMAGE_PROCESSING_IMAGECREATION_H