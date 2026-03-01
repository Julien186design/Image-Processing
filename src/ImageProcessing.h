#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H
#include "Image.h"
#include "ProcessingConfig.h"
#include "TransformationsConfig.h"

#include <string>
#include <vector>
#include <functional>
#include <cstring>
#include <format>

using GenericTransformationFunc = std::function<void(Image&, int, const std::vector<int>&)>;
using GenericTransformationFuncWithColorNuances = std::function<void(Image&, int, int, int, const std::vector<int>&)>;


class ImageBuffer {
private:
    Image buffer;

    struct ChannelIndices {
        int max;
        int mid;
        int min;
    };

public:
    ImageBuffer(const int w, const int h, const int channels) : buffer(w, h, channels) {}

    void resetFrom(const Image& source) const {
        assert(buffer.size == source.size);
        std::memcpy(buffer.data, source.data, buffer.size);
    }


    Image& get() { return buffer; }

    void saveAs(const char* path) {
        buffer.write(path);
    }

    static void clear() {
        // Force la libération si Image a une méthode dédiée
        // Sinon, cette méthode ne fait rien (destructeur s'en charge)
        // buffer.release();  // Si Image a cette méthode
    }
};


struct WeightedColors {
    uint8_t r_third, g_third, b_third;
    uint8_t r_half, g_half, b_half;
    uint8_t r_full, g_full, b_full;
};

inline WeightedColors calculateWeightedColors(
    const std::vector<float>& weightOfRGB) {
    return {
        static_cast<uint8_t>(weightOfRGB[0] * SimpleColors::ONE_THIRD),
        static_cast<uint8_t>(weightOfRGB[1] * SimpleColors::ONE_THIRD),
        static_cast<uint8_t>(weightOfRGB[2] * SimpleColors::ONE_THIRD),

        static_cast<uint8_t>(weightOfRGB[0] * SimpleColors::HALF),
        static_cast<uint8_t>(weightOfRGB[1] * SimpleColors::HALF),
        static_cast<uint8_t>(weightOfRGB[2] * SimpleColors::HALF),

        static_cast<uint8_t>(weightOfRGB[0] * SimpleColors::FULL),
        static_cast<uint8_t>(weightOfRGB[1] * SimpleColors::FULL),
        static_cast<uint8_t>(weightOfRGB[2] * SimpleColors::FULL)
    };
}


inline std::vector<std::vector<float>> generateColorConfigs(
    const std::vector<float>& weightOfRGB,
    const bool binaryOnly = false)         // if true, only {0,1}^3 \ {0,0,0} configs
{
    const int n = static_cast<int>((weightOfRGB[1] - weightOfRGB[0]) / weightOfRGB[2]);
    const int total_configs = (n + 1) * (n + 1) * (n + 1);

    std::vector<std::vector<float>> configs;
    configs.reserve(binaryOnly ? 7 : total_configs);

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            for (int k = 0; k <= n; ++k) {
                // When binaryOnly: keep only indices in {0,1}, skip (0,0,0)
                if (binaryOnly) {
                    if (i < n && i > 0 || j < n && j > 0 || k < n && k > 0) continue;
                    if ((i | j | k) == 0) continue;        // exclut {0,0,0}
                }
                configs.push_back({
                    weightOfRGB[0] + k * weightOfRGB[2],
                    weightOfRGB[0] + j * weightOfRGB[2],
                    weightOfRGB[0] + i * weightOfRGB[2]
                });
            }
        }
    }

    return configs;
}

struct OneColorPipeline {
    const std::vector<std::vector<float>> configs;
    const bool average;
    const bool integerMode;

    explicit OneColorPipeline(
        const std::vector<float>& weightOfRGB,
        const bool average,
        const bool binaryOnly = false,
        const bool integerMode = false)
        : configs(generateColorConfigs(weightOfRGB, binaryOnly))
        , average(average)
        , integerMode(integerMode)
    {}

    struct ConfigParams {
        uint8_t r_third, g_third, b_third;
        uint8_t r_half,  g_half,  b_half;
        uint8_t r_full,  g_full,  b_full;
        std::string weightedColors;
    };

    // Pre-compute the color constants for a given config index
    [[nodiscard]] ConfigParams buildParams(const size_t configIdx) const {
        const auto& config = configs[configIdx];
        const auto [r_third, g_third, b_third,
                    r_half,  g_half,  b_half,
                    r_full,  g_full,  b_full] = calculateWeightedColors(config);
        return { r_third, g_third, b_third,
                 r_half,  g_half,  b_half,
                 r_full,  g_full,  b_full,
                 OutputPathBuilder::writingWeightedColors(config, integerMode) };
    }

    // Apply without_average transform to an Image in-place
    static void applyWithoutAverage(Image& img, const int tolerance, const ConfigParams& p) {
        img.simplify_to_dominant_color_combinations_without_average(
            tolerance,
            p.r_third, p.g_third, p.b_third,
            p.r_half,  p.g_half,  p.b_half,
            p.r_full,  p.g_full,  p.b_full
        );
    }

    // Apply with_average transform to an Image in-place
    static void applyWithAverage(Image& img, const int tolerance,
                                 const std::vector<float>& config) {
        img.simplify_to_dominant_color_combinations_with_average(tolerance, config);
    }

    [[nodiscard]] size_t configCount() const { return configs.size(); }
};

// Processes a single reversal at a fixed proportion value (used for the p=0.5 edge case).
// Defined inline to avoid ODR violations across translation units.
inline void reverse_at_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const float p
) {
    for (size_t i = 0; i < reversal_step_by_step_entries.size(); ++i) {
        const auto& [suffix, output_dir] = reversal_step_by_step_entries[i];
        const bool below = (i == 0);

        ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);
        modified.resetFrom(baseImage);
        modified.get().reverse_by_proportion(p, below);

        std::string outputPath = OutputPathBuilder::reverse(output_dir, baseName, suffix, p);
        modified.saveAs(outputPath.c_str());
    }
}

template<typename ApplyFunc, typename PathFunc>
void run_transformations_by_proportion(
    const Image& baseImage,
    const std::string& baseName,
    const PropRange& proportions,
    const std::vector<TransformationEntry>& entries,
    ApplyFunc apply,
    PathFunc buildPath,
    const std::vector<int>* colorNuances = nullptr
) {
    const int num_threads = computeNumThreads();
    const size_t num_props =
        static_cast<size_t>((proportions.stop - proportions.start) / proportions.step) + 1;

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
shared(baseImage, baseName, entries, colorNuances, apply, buildPath, process_one) \
firstprivate(num_props, proportions)
    for (size_t p_idx = 0; p_idx < num_props; ++p_idx) {
        const float p = proportions.start + p_idx * proportions.step;

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

void processImageTransforms(
    const std::string& baseName ,
    const std::string& inputPath,
    const PropRange& proportions,
    const std::vector<int>& colorNuances,
    const std::vector<int>& rectangles,
    const std::vector<int>&  tolerance,
    const std::vector<float>& weightOfRGB
);

#endif
