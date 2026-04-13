#ifndef IMAGE_PROCESSING_COLORCONFIG_H
#define IMAGE_PROCESSING_COLORCONFIG_H

#include <mutex>

#include "Image.h"
#include "ProcessingConfig.h"
#include <ostream>
#include <vector>

// --- Hilbert 3D mapping ---
// Converts (x,y,z) in [0, 2^bits - 1] to a Hilbert index.
// Based on John Skilling's algorithm (compact, branch-light).

inline auto hilbert3D(uint32_t x, uint32_t y, uint32_t z, const uint32_t bits) -> uint32_t
{
    uint32_t index = 0;
    uint32_t mask = 1U << (bits - 1);

    for (uint32_t i = 0; i < bits; ++i) {
        uint32_t h = 0;

        // Extract current bit of each coordinate
        const uint32_t xi = ((x & mask) != 0U) ? 1U : 0U;
        const uint32_t yi = ((y & mask) != 0) ? 1U : 0U;
        const uint32_t zi = ((z & mask) != 0U) ? 1U : 0U;

        // Interleave bits (Gray code style ordering)
        h = (xi << 2U) | (yi << 1U) | zi;

        index = (index << 3U) | h;

        // Rotate / reflect (key step for Hilbert continuity)
        if (yi == 0) {
            if (zi == 1) {
                x = (~x) & ((1U << bits) - 1);
                y = (~y) & ((1U << bits) - 1);
            }
            std::swap(x, z);
        }

        mask >>= 1U;
    }

    return index;
}



inline auto generateColorConfigs(
    const bool binaryOnly) -> std::vector<std::vector<float>>
{
    constexpr int size = parameters::iterationsRGB;

    std::vector<std::vector<float>> configs;
    configs.reserve(size * size * size);

    auto toWeight = [](const int idx) -> float {
        return std::get<0>(parameters::weightOfRGB) +
               (static_cast<float>(idx) * std::get<2>(parameters::weightOfRGB));
    };

    if (binaryOnly) {
        const std::array<std::array<float,3>, 7> corners = {{
            {1, 0, 0},
            {0, 1, 0},
            {1, 1, 0},
            {0, 0, 1},
            {1, 0, 1},
            {0, 1, 1},
            {1, 1, 1},
        }};
        for (const auto& [k, j, i] : corners) { 
            configs.push_back({ k, j, i });
}
        std::cout << "binaryOnly: [";
        for (size_t i = 0; i < configs.size(); ++i) {
            std::cout << configs.at(i);
            if (i != configs.size() - 1) { std::cout << ", ";
            }
        }
        std::cout << "]" << '\n';
        return configs;
    }

    for (int i = 0; i < parameters::iterationsRGB; ++i) {
        configs.push_back({ 1, 0, toWeight(i) });
    }

    for (int i = 1; i < parameters::iterationsRGB; ++i) {
        configs.push_back({ 1, toWeight(i), 1 });
    }
    std::cout << "Complete: [\n";
    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << i << " - " << configs.at(i);
        if (i != configs.size() - 1) { std::cout << "\n";
        }
    }
    std::cout << "]" << '\n';
    return configs;
}

struct OneColorPipeline {
    const std::vector<std::vector<float>> configs;
    const std::vector<float> tValues;

    static OneColorPipeline forStatic()    { return OneColorPipeline(true); }
    static OneColorPipeline forStreaming() { return OneColorPipeline(false);  }

    struct ConfigParams {
        std::string weightedColors;
    };

    [[nodiscard]] static size_t outputCount() {
        return parameters::numProportionSteps + 1;
    }

    [[nodiscard]] ConfigParams buildParams(const size_t configIdx) const {
        return { OutputPathBuilder::writingWeightedColors(configs.at(configIdx)) };
    }

    [[nodiscard]] size_t configCount() const { return configs.size(); }
    [[nodiscard]] size_t passCount()   const { return tValues.size(); }
    [[nodiscard]] const std::vector<float>& getTValues() const { return tValues; }

    // Used by oneColorTransformations (static images):
    // passes = parameters::numProportionSteps (empty span triggers that path).
    [[nodiscard]] auto applyStatic(
        const Image& img, const int tolerance, const size_t configIdx
    ) const -> std::vector<Image> {
        std::vector<Image> out;
        out.reserve(outputCount());
        img.simplify_to_dominant_color_combinations(
            tolerance, &configs.at(configIdx), {},
            [&out](Image&& result) {
                out.push_back(std::move(result));
                return true; // always continue
            }
        );
        return out;
    }

    [[nodiscard]] std::vector<Image> applyStreaming(
        const Image& img, const int tolerance, const size_t configIdx
    ) const {
        std::vector<Image> out;
        out.reserve(tValues.size());
        img.simplify_to_dominant_color_combinations(
            tolerance, &configs.at(configIdx), tValues,
            [&out](Image&& result) {
                out.push_back(std::move(result));
                return true;
            }
        );
        return out;
    }

private:
    mutable std::once_flag tValuesPrinted;

    explicit OneColorPipeline(const bool binaryOnly)
        : configs(generateColorConfigs(binaryOnly))
        , tValues(buildTValues())
    {}

    static std::vector<float> buildTValues() {
        std::vector<float> values;
        const auto [from, to, step] = parameters::passesRGB;
        const int steps  = static_cast<int>((to - from) / step);

        for (int i = 0; i <= steps; ++i) {
            const float tVal = from + (static_cast<float>(i) * step);
            values.push_back(std::clamp(tVal, std::min(from, to), std::max(from, to)));
        }
        return values;
    }
};

#endif //IMAGE_PROCESSING_COLORCONFIG_H
