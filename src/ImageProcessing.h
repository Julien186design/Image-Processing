#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "Image.h"

#include <string>
#include <vector>
#include <functional>
#include <cstring>
#include <format>

class EdgeDetectorPipeline {
    const int width;
    const int height;
    const size_t imgSize;
    const double threshold;

    // 3x3 Gaussian kernel (normalized by 1/16)
    static constexpr double inv16 = 1.0 / 16.0;
    double gaussKernel[9] = {
        inv16, 2 * inv16, inv16,
        2 * inv16, 4 * inv16, 2 * inv16,
        inv16, 2 * inv16, inv16
    };

    // Internal working buffers (allocated once)
    std::vector<double> blurData;
    std::vector<double> tx, ty;
    std::vector<double> gx, gy;
    std::vector<double> g, theta;
    std::vector<uint8_t> outputRGB;

    // Persistent grayscale image buffer (avoids per-frame allocation)
    Image tempImg;

public:

    // Constructor: allocate all buffers once
    EdgeDetectorPipeline(const int w, const int h, const double thresh = 0.09)
        : width(w),
          height(h),
          imgSize(static_cast<size_t>(w) * h),
          threshold(thresh),
          blurData(imgSize),
          tx(imgSize),
          ty(imgSize),
          gx(imgSize),
          gy(imgSize),
          g(imgSize),
          outputRGB(imgSize * 3),
          tempImg(w, h, 1)
    {}

    const std::vector<uint8_t>& process(const uint8_t* grayData)
    {
        // Copy grayscale input into persistent image buffer
        std::memcpy(tempImg.data, grayData, imgSize);

        // Apply Gaussian blur in-place
        tempImg.convolve_linear(0, 3, 3, gaussKernel, 1, 1);

        // -------------------------
        // Stage 1: Normalize to double
        // -------------------------
        #pragma omp parallel for schedule(static) default(none) \
    		shared(blurData)
        for (size_t k = 0; k < imgSize; ++k)
            blurData[k] = tempImg.data[k] / 255.0;

        // -------------------------
        // Stage 2: Horizontal derivative pass
        // -------------------------
		#pragma omp parallel for collapse(2) schedule(static) default(none) \
			shared(blurData, tx, ty)
        for (int r = 0; r < height; ++r)
        {
            for (int c = 1; c < width - 1; ++c)
            {
                const size_t idx = static_cast<size_t>(r) * width + c;

                tx[idx] = blurData[idx + 1] - blurData[idx - 1];

                ty[idx] = 47.0 * blurData[idx + 1]
                        + 162.0 * blurData[idx]
                        + 47.0 * blurData[idx - 1];
            }
        }

        // -------------------------
        // Stage 3: Vertical derivative pass
        // -------------------------
		#pragma omp parallel for collapse(2) schedule(static) default(none) \
			shared(tx, ty, gx, gy)
        for (int c = 1; c < width - 1; ++c)
        {
            for (int r = 1; r < height - 1; ++r)
            {
                const size_t idx = static_cast<size_t>(r) * width + c;

                gx[idx] = 47.0 * tx[(r + 1) * width + c]
                        + 162.0 * tx[r * width + c]
                        + 47.0 * tx[(r - 1) * width + c];

                gy[idx] = ty[(r + 1) * width + c]
                        - ty[(r - 1) * width + c];
            }
        }

        // -------------------------
        // Stage 4: Gradient magnitude and orientation
        // -------------------------
		#pragma omp parallel for schedule(static) default(none) \
			shared(g, gx, gy)
        for (size_t k = 0; k < imgSize; ++k)
        {
            g[k] = std::sqrt(gx[k] * gx[k] + gy[k] * gy[k]);
        }

        // -------------------------
        // Stage 5: Parallel min/max reduction (portable version)
        // -------------------------
        double mx = -std::numeric_limits<double>::infinity();
        double mn =  std::numeric_limits<double>::infinity();

		#pragma omp parallel default(none) shared(g, mx, mn)
        {
            double local_max = -std::numeric_limits<double>::infinity();
            double local_min =  std::numeric_limits<double>::infinity();

            #pragma omp for nowait
            for (size_t k = 0; k < imgSize; ++k)
            {
                if (g[k] > local_max) local_max = g[k];
                if (g[k] < local_min) local_min = g[k];
            }

            #pragma omp critical
            {
                if (local_max > mx) mx = local_max;
                if (local_min < mn) mn = local_min;
            }
        }

        // -------------------------
        // Stage 6: HSL → RGB mapping
        // -------------------------
		#pragma omp parallel for schedule(static) default(none) \
			shared(g, outputRGB) firstprivate(mx, mn)
        for (size_t k = 0; k < imgSize; ++k)
        {
        	const double h = std::atan2(gy[k], gx[k]) * 180.0 / M_PI + 180.0;
            double v = (mx == mn) ? 0.0 : (g[k] - mn) / (mx - mn);
            v = (v > threshold) ? v : 0.0;

            const double s = v;
            const double l = v;

            const double c = (1.0 - std::fabs(2.0 * l - 1.0)) * s;
            const double x = c * (1.0 - std::fabs(std::fmod(h / 60.0, 2.0) - 1.0));
            const double m = l - c / 2.0;

            double rt = 0.0, gt = 0.0, bt = 0.0;

            if      (h < 60.0)  { rt = c; gt = x; }
            else if (h < 120.0) { rt = x; gt = c; }
            else if (h < 180.0) { gt = c; bt = x; }
            else if (h < 240.0) { gt = x; bt = c; }
            else if (h < 300.0) { bt = c; rt = x; }
            else                { bt = x; rt = c; }

            outputRGB[k * 3    ] = static_cast<uint8_t>(255.0 * (rt + m));
            outputRGB[k * 3 + 1] = static_cast<uint8_t>(255.0 * (gt + m));
            outputRGB[k * 3 + 2] = static_cast<uint8_t>(255.0 * (bt + m));
        }

        return outputRGB;
    }
};

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
    const bool binaryOnly = false)         // if true, only {0,1}^3 \ {0,0,0} configs
{
    constexpr auto n = static_cast<int>((parameters::weightOfRGB[1] - parameters::weightOfRGB[0]) /
                                        parameters::weightOfRGB[2]);
    constexpr int total_configs = (n + 1) * (n + 1) * (n + 1);

    std::vector<std::vector<float>> configs;
    configs.reserve(binaryOnly ? 7 : total_configs);

    bool reverse_order_of_k = false;

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (reverse_order_of_k) {
                for (int k = 0; k <= n; ++k) {
                    // When binaryOnly: keep only indices in {0,1}, skip (0,0,0)
                    if (binaryOnly) {
                        if (i < n && i > 0 || j < n && j > 0 || k < n && k > 0) continue;
                        if ((i | j | k) == 0) continue;        // exclut {0,0,0}
                    }
                    configs.push_back({
                        parameters::weightOfRGB[0] + static_cast<float>(k) * parameters::weightOfRGB[2],
                        parameters::weightOfRGB[0] + static_cast<float>(j) * parameters::weightOfRGB[2],
                        parameters::weightOfRGB[0] + static_cast<float>(i) * parameters::weightOfRGB[2]
                    });
                }
                reverse_order_of_k = !reverse_order_of_k;
            } else {
                for (int k = n; k >= 0; --k) {
                    // When binaryOnly: keep only indices in {0,1}, skip (0,0,0)
                    if (binaryOnly) {
                        if (i < n && i > 0 || j < n && j > 0 || k < n && k > 0) continue;
                        if ((i | j | k) == 0) continue;        // exclut {0,0,0}
                    }
                    configs.push_back({
                        parameters::weightOfRGB[0] + static_cast<float>(k) * parameters::weightOfRGB[2],
                        parameters::weightOfRGB[0] + static_cast<float>(j) * parameters::weightOfRGB[2],
                        parameters::weightOfRGB[0] + static_cast<float>(i) * parameters::weightOfRGB[2]
                    });
                }
                reverse_order_of_k = !reverse_order_of_k;
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
        const bool binaryOnly  = false,
        const bool integerMode = false)
        : configs(generateColorConfigs(binaryOnly))
        , average(parameters::average)
        , integerMode(integerMode)
    {}

    struct ConfigParams {
        uint8_t r_third, g_third, b_third;
        uint8_t r_half,  g_half,  b_half;
        uint8_t r_full,  g_full,  b_full;
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

    void apply(Image& img, const int tolerance,
               const size_t configIdx, const ConfigParams& p) const {
        if (average) {
            img.simplify_to_dominant_color_combinations_with_average(tolerance, configs[configIdx]);
        } else {
            img.simplify_to_dominant_color_combinations_without_average(
                tolerance,
                p.r_third, p.g_third, p.b_third,
                p.r_half,  p.g_half,  p.b_half,
                p.r_full,  p.g_full,  p.b_full
            );
        }
    }
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


#endif
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
