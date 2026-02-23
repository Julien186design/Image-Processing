#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H
#include "Image.h"
#include "TransformationsConfig.h"

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <cstring>
#include <algorithm> // for std::min et std::max
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

class OutputPathBuilder {
public:
    static std::string buildStandard(
        const std::string& outputDir,
        const std::string& baseName,
        const std::string& transformationType,
        const std::string& suffix,
        const int threshold
    ) {
        std::ostringstream oss;
        oss << outputDir << baseName << transformationType << suffix
            << " " << threshold;
        return oss.str();
    }

    static std::string build120(
        const std::string& folder120,
        const std::string& baseName,
        const std::string& transformationType,
        const std::string& suffix
    ) {
        std::ostringstream oss;
        oss << folder120 << baseName << transformationType << suffix;
        return oss.str();
    }
};

struct WeightedColors {
    uint8_t r_third, g_third, b_third;
    uint8_t r_half, g_half, b_half;
    uint8_t r_full, g_full, b_full;
};

inline std::vector<int> genererRectanglesInDiagonal(const int fraction) {
    if (fraction <= 0) return {};
    const int taille = 1 << fraction;  // 2^fraction via bit-shift
    const int pas = taille + 1;

    std::vector<int> vector;
    vector.reserve(taille);

    vector.push_back(-1);
    for (int i = 0; i < taille; ++i) {
        vector.push_back(i * pas);
    }
    return vector;
}


inline std::vector<int> range_to_vector(const std::vector<int>& input) {
    if (input.size() != 2) {
        return {}; // Retourne un vecteur vide si l'entrée n'a pas 2 éléments
    }

    const int start = std::min(input[0], input[1]);
    const int end = std::max(input[0], input[1]);

    std::vector<int> vector;
    for (int i = start; i <= end; ++i) {
        vector.push_back(i);
    }
    return vector;
}



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

inline std::string makeWeightedColors(const std::vector<float>& weightOfRGB) {
    std::string s;
    s.reserve(32); // to avoid several allocations

    s += ' ';
    s += std::format("{:.2f}", weightOfRGB[0]);
    s += '-';
    s += std::format("{:.2f}", weightOfRGB[1]);
    s += '-';
    s += std::format("{:.2f}", weightOfRGB[2]);

    return s;
}


inline std::vector<std::vector<float>> generateColorConfigs(const std::vector<float> &weightOfRGB) {
    const int n = static_cast<int>((weightOfRGB[1] - weightOfRGB[0]) / weightOfRGB[2]);
    const int total_configs = (n + 1) * (n + 1) * (n + 1);

    std::vector<std::vector<float>> configs;
    configs.reserve(total_configs);

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            for (int k = 0; k <= n; ++k) {
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

    explicit OneColorPipeline(const std::vector<float>& weightOfRGB, const bool average)
        : configs(generateColorConfigs(weightOfRGB)), average(average) {}

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
                 makeWeightedColors(config) };
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

void processImageTransforms(
    const std::string& baseName ,
    const std::string& inputPath,
    const std::vector<float> &proportions,
    const std::vector<int>& colorNuances,
    int fraction,
    std::vector<int>& rectanglesToModify,
    const std::vector<int>&  tolerance,
    const std::vector<float>& weightOfRGB,
    bool severalColorsByProportion,
    bool totalReversal,
    bool partial,
    bool partialInDiagonal,
    bool oneColor,
    bool average
);

#endif
