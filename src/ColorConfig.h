#ifndef IMAGE_PROCESSING_COLORCONFIG_H
#define IMAGE_PROCESSING_COLORCONFIG_H

#include "Image.h"
#include "ProcessingConfig.h"

struct WeightedColors {
    std::vector<uint8_t> r_third, g_third, b_third;
    std::vector<uint8_t> r_half, g_half, b_half;
    std::vector<uint8_t> r_full, g_full, b_full;
};

inline WeightedColors calculateWeightedColors(
    const std::vector<float>& weightOfRGB) {
    if (weightOfRGB.size() != 3) {
        throw std::invalid_argument("weightOfRGB must contain exactly 3 elements (R, G, B).");
    }

    auto clampAndCast = [](const float value) {
        return static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, value)));
    };
    return {
        {clampAndCast(weightOfRGB[0] * SimpleColors::ONE_THIRD)},
        {clampAndCast(weightOfRGB[1] * SimpleColors::ONE_THIRD)},
        {clampAndCast(weightOfRGB[2] * SimpleColors::ONE_THIRD)},

        {clampAndCast(weightOfRGB[0] * SimpleColors::HALF)},
        {clampAndCast(weightOfRGB[1] * SimpleColors::HALF)},
        {clampAndCast(weightOfRGB[2] * SimpleColors::HALF)},

        {clampAndCast(weightOfRGB[0] * SimpleColors::FULL)},
        {clampAndCast(weightOfRGB[1] * SimpleColors::FULL)},
        {clampAndCast(weightOfRGB[2] * SimpleColors::FULL)}
    };
}


inline std::vector<std::vector<float>> generateColorConfigs(
    const bool binaryOnly = false)  // if true, only {0,1}^3 \ {0,0,0} configs
{
    constexpr int n = static_cast<int>(
        (parameters::weightOfRGB[1] - parameters::weightOfRGB[0]) /
         parameters::weightOfRGB[2]);

    constexpr int total_configs = (n + 1) * (n + 1) * (n + 1);

    std::vector<std::vector<float>> configs;
    configs.reserve(binaryOnly ? 7 : total_configs);

    // Lambda to convert a discrete index to its float weight
    auto toWeight = [](const int idx) -> float {
        return parameters::weightOfRGB[0] +
               static_cast<float>(idx) * parameters::weightOfRGB[2];
    };

    bool reverse_j = false;

    for (int i = 0; i <= n; ++i) {
        bool reverse_k = false;

        // Snake-order on j: alternate direction each row
        const int j_start = reverse_j ? n : 0;
        const int j_end   = reverse_j ? 0 : n;
        const int j_step  = reverse_j ? -1 : 1;

        for (int j = j_start; j != j_end + j_step; j += j_step) {
            // Snake-order on k: alternate direction each row
            const int k_start = reverse_k ? n : 0;
            const int k_end   = reverse_k ? 0 : n;
            const int k_step  = reverse_k ? -1 : 1;

            for (int k = k_start; k != k_end + k_step; k += k_step) {
                // binaryOnly: keep only corner indices {0, n}^3, exclude {0,0,0}
                if (binaryOnly) {
                    if ((i > 0 && i < n) || (j > 0 && j < n) || (k > 0 && k < n))
                        continue;
                    if ((i | j | k) == 0)
                        continue;
                }
                configs.push_back({ toWeight(k), toWeight(j), toWeight(i) });
            }
            reverse_k = !reverse_k;
        }
        reverse_j = !reverse_j;
    }

    return configs;
}

struct OneColorPipeline {
    const std::vector<std::vector<float>> configs;
    const bool integerMode;

    explicit OneColorPipeline(
        const bool binaryOnly  = false,
        const bool integerMode = false)
        : configs(generateColorConfigs(binaryOnly))
        , integerMode(integerMode)
    {}

    struct ConfigParams {
        std::vector<uint8_t> r_third, g_third, b_third;
        std::vector<uint8_t> r_half,  g_half,  b_half;
        std::vector<uint8_t> r_full,  g_full,  b_full;
        std::string weightedColors;
    };

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

    [[nodiscard]] size_t configCount() const { return configs.size(); }

    [[nodiscard]] std::vector<Image> apply(const Image& img, const int tolerance,
               const size_t configIdx) const {

        return img.simplify_to_dominant_color_combinations(
            tolerance,
            &configs[configIdx]
        );
    }
};

#endif //IMAGE_PROCESSING_COLORCONFIG_H