#ifndef IMAGE_PROCESSING_COLORCONFIG_H
#define IMAGE_PROCESSING_COLORCONFIG_H

#include <mutex>

#include "Image.h"
#include "ProcessingConfig.h"

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
        h = (xi << 2) | (yi << 1) | zi;

        index = (index << 3) | h;

        // Rotate / reflect (key step for Hilbert continuity)
        if (yi == 0) {
            if (zi == 1) {
                x = (~x) & ((1U << bits) - 1);
                y = (~y) & ((1U << bits) - 1);
            }
            std::swap(x, z);
        }

        mask >>= 1;
    }

    return index;
}

inline auto generateColorConfigs(
    const bool binaryOnly = false) -> std::vector<std::vector<float>>
{
    constexpr int size = parameters::iterationsRGB + 1;

    std::vector<std::vector<float>> configs;
    configs.reserve(size * size * size);

    auto toWeight = [](const int idx) -> float {
        return parameters::weightOfRGB[0] +
               (static_cast<float>(idx) * parameters::weightOfRGB[2]);
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
        for (const auto& [k, j, i] : corners) { // NOLINT(llvmlibc-callee-namespace)
            configs.push_back({ k, j, i });
}
        return configs;
    }

    // --- Step 1: build full grid with integer coordinates ---
    struct Node {
        uint32_t x, y, z;
        uint32_t hilbert;
    };

    std::vector<Node> nodes;
    nodes.reserve(size * size * size);

    // Determine number of bits needed
    uint32_t bits = 0;
    while ((1U << bits) < static_cast<uint32_t>(size)) { ++bits;
}

    for (uint32_t i = 0; i <= static_cast<uint32_t>(parameters::iterationsRGB); ++i) {
        for (uint32_t j = 0; j <= static_cast<uint32_t>(parameters::iterationsRGB); ++j) {
            for (uint32_t k = 0; k <= static_cast<uint32_t>(parameters::iterationsRGB); ++k) {

                // Normalize to power-of-two grid
                const uint32_t x = k;
                const uint32_t y = j;
                const uint32_t z = i;

                nodes.push_back({
                    x, y, z,
                    hilbert3D(x, y, z, bits) // NOLINT(llvmlibc-callee-namespace)
                });
            }
        }
    }

    // --- Step 2: sort by Hilbert index ---
    std::ranges::sort(nodes,
                      [](const Node& a, const Node& b) {
                          return a.hilbert < b.hilbert;
                      });

    // --- Step 3: convert to float configs ---
    for (const auto& node : nodes) {
        configs.push_back({
            toWeight(static_cast<int>(node.x)),
            toWeight(static_cast<int>(node.y)),
            toWeight(static_cast<int>(node.z))
        });
    }

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
        return { OutputPathBuilder::writingWeightedColors(configs[configIdx]) };
    }

    [[nodiscard]] size_t configCount() const { return configs.size(); }
    [[nodiscard]] size_t passCount()   const { return tValues.size(); }

    // Used by oneColorTransformations (static images):
    // passes = parameters::numProportionSteps (empty span triggers that path).
    [[nodiscard]] auto applyStatic(
        const Image& img,
        const int tolerance,
        const size_t configIdx
    ) const -> std::vector<Image> {
        return img.simplify_to_dominant_color_combinations(
            tolerance,
            &configs[configIdx],
            {}
        );
    }

    [[nodiscard]] std::vector<Image> applyStreaming(
        const Image& img,
        const int tolerance,
        const size_t configIdx
    ) const {
        std::call_once(tValuesPrinted, [this]() {
            std::cout << "tValues: [";
            for (size_t i = 0; i < tValues.size(); ++i) {
                std::cout << tValues[i];
                if (i != tValues.size() - 1) { std::cout << ", ";
}
            }
            std::cout << "]" << std::endl;
        });

        return img.simplify_to_dominant_color_combinations(
            tolerance,
            &configs[configIdx],
            tValues
        );
    }

private:
    mutable std::once_flag tValuesPrinted;

    explicit OneColorPipeline(const bool binaryOnly)
        : configs(generateColorConfigs(binaryOnly))
        , tValues(buildTValues())
    {}

    static std::vector<float> buildTValues() {
        std::vector<float> values;
        constexpr float from = parameters::passesRGB[0];
        constexpr float to   = parameters::passesRGB[1];
        constexpr float step = parameters::passesRGB[2];
        constexpr int steps  = static_cast<int>((to - from) / step);

        for (int i = 0; i <= steps; ++i) {
            const float tVal = from + static_cast<float>(i) * step;
            values.push_back(std::clamp(tVal, std::min(from, to), std::max(from, to)));
        }
        return values;
    }
};

#endif //IMAGE_PROCESSING_COLORCONFIG_H
