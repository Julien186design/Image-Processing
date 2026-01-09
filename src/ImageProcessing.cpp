#include "ImageProcessing.h"
#include "TransformationsConfig.h"
#include <vector>
#include <functional>
#include <string>
#include <cstring>

// ============================================================================
// TYPES ET STRUCTURES
// ============================================================================

using GenericTransformationFunc = std::function<void(Image&, int, const std::vector<int>&)>;

struct TransformationContext {
    const Image& baseImage;
    const std::string& baseName;
    int firstThreshold;
    int lastThreshold;
    int step;
    bool saveAt120;
    std::string folder120;
};

// ============================================================================
// UTILITAIRES
// ============================================================================

class ImageBuffer {
private:
    Image buffer;

public:
    ImageBuffer(int w, int h, int channels) : buffer(w, h, channels) {}

    void resetFrom(const Image& source) {
        std::memcpy(buffer.data, source.data, source.size);
    }

    Image& get() { return buffer; }

    void saveAs(const char* path) {
        buffer.write(path);
    }
};

class OutputPathBuilder {
public:
    static std::string buildStandard(
        const std::string& outputDir,
        const std::string& baseName,
        const std::string& transformationType,
        const std::string& suffix,
        int threshold
    ) {
        return outputDir + baseName + transformationType + suffix + " " + std::to_string(threshold) + ".png";
    }

    static std::string build120(
        const std::string& folder120,
        const std::string& baseName,
        const std::string& transformationType,
        const std::string& suffix
    ) {
        return folder120 + baseName + transformationType + suffix + " 120.png";
    }
};

// ============================================================================
// WRAPPERS POUR TYPES DE TRANSFORMATIONS
// ============================================================================

std::vector<GenericTransformationFunc> wrapSimpleTransforms(
    const std::vector<TransformationFunc>& transforms
) {
    std::vector<GenericTransformationFunc> wrapped;
    wrapped.reserve(transforms.size());

    for (const auto& transform : transforms) {
        wrapped.emplace_back([transform](Image& img, int t, const std::vector<int>&) {
            transform(img, t);
        });
    }

    return wrapped;
}

std::vector<GenericTransformationFunc> wrapAlternatingTransforms(
    const std::vector<AlternatingTransformation>& transforms
) {
    std::vector<GenericTransformationFunc> wrapped;
    wrapped.reserve(transforms.size());

    for (const auto& transform : transforms) {
        wrapped.emplace_back([transform](Image& img, int, const std::vector<int>& params) {
            // params[0] = firstThreshold, params[1] = lastThreshold, params[2] = step
            transform(img, params[0], params[1], params[2]);
        });
    }

    return wrapped;
}

std::vector<GenericTransformationFunc> wrapPartialTransforms(
    const std::vector<PartialTransformationFunc>& transforms
) {
    std::vector<GenericTransformationFunc> wrapped;
    wrapped.reserve(transforms.size());

    for (const auto& transform : transforms) {
        wrapped.emplace_back([transform](Image& img, int t, const std::vector<int>& params) {
            // params[0] = fraction, params[1..n] = rectanglesToModify
            int fraction = params[0];
            std::vector<int> rectangles(params.begin() + 1, params.end());
            transform(img, t, fraction, rectangles);
        });
    }

    return wrapped;
}

// ============================================================================
// FONCTION GÉNÉRIQUE CENTRALE
// ============================================================================

void applyAndSaveGenericTransformations(
    const Image& baseImage,
    const std::vector<GenericTransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const std::string& transformationType,
    int threshold,
    int lastThreshold,
    int step,
    bool saveAt120,
    const std::string& folder120,
    const std::vector<int>& additionalParams
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int currentThreshold = threshold; currentThreshold <= lastThreshold; currentThreshold += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {
            modified.resetFrom(baseImage);

            transforms[i](modified.get(), currentThreshold, additionalParams);

            std::string outputPath = OutputPathBuilder::buildStandard(
                outputDirs[i],
                baseName,
                transformationType,
                suffixes[i],
                currentThreshold
            );
            modified.saveAs(outputPath.c_str());

            if (saveAt120 && currentThreshold == 120) {
                std::string specialPath = OutputPathBuilder::build120(
                    folder120,
                    baseName,
                    transformationType,
                    suffixes[i]
                );
                modified.saveAs(specialPath.c_str());
            }
        }
    }
}

// ============================================================================
// FONCTIONS DE COMPATIBILITÉ (LEGACY)
// ============================================================================

void applyTransformationsWrapper(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    bool saveAt120,
    const std::string& folder120
) {
    applyAndSaveGenericTransformations(
        baseImage,
        wrapSimpleTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " - ",
        threshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        {}
    );
}

void applyAndSaveTransformations(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    bool saveAt120,
    const std::string& folder120
) {
    applyTransformationsWrapper(
        baseImage,
        transforms,
        outputDirs,
        suffixes,
        baseName,
        threshold,
        lastThreshold,
        step,
        saveAt120,
        folder120
    );
}

void applyAndSaveReversalTransformations(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    bool saveAt120,
    const std::string& folder120
) {
    applyAndSaveGenericTransformations(
        baseImage,
        wrapSimpleTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " Reversal - ",
        threshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        {}
    );
}

void applyAndSaveAlternatingTransformations(
    const Image& baseImage,
    const std::vector<AlternatingTransformation>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int firstThreshold,
    int lastThreshold,
    int step,
    bool saveAt120,
    const std::string& folder120
) {
    std::vector<int> additionalParams = {firstThreshold, lastThreshold, step};

    applyAndSaveGenericTransformations(
        baseImage,
        wrapAlternatingTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " Alternating - ",
        firstThreshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        additionalParams
    );
}

void applyAndSavePartialTransformations(
    const Image& baseImage,
    const std::vector<PartialTransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    int threshold,
    int lastThreshold,
    int step,
    int fraction,
    const std::vector<int>& rectanglesToModify
) {
    std::vector<int> partialParams = {fraction};
    partialParams.insert(partialParams.end(), rectanglesToModify.begin(), rectanglesToModify.end());

    applyAndSaveGenericTransformations(
        baseImage,
        wrapPartialTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " - ",
        threshold,
        lastThreshold,
        step,
        false,
        "",
        partialParams
    );
}

// ============================================================================
// TRANSFORMATIONS SPÉCIALISÉES
// ============================================================================

void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& tolerance
) {
    const int start = tolerance[0];
    const int end   = tolerance[1];
    const int step  = tolerance[2];
    bool average = false;

    const std::string oneColorFolder = "Output/One Color";

    for (int t = start; t <= end; t += step) {
        for (int i = 0; i < 2; ++i) {
            Image oneColorImage = baseImage;

            oneColorImage.simplify_to_dominant_color_combinations(t, average);

            std::string outputPath = oneColorFolder + baseName + " - Average " + std::to_string(average) +
                                    " - Tolerance " + std::to_string(t) + ".png";
            oneColorImage.write(outputPath.c_str());

            average = !average;
        }
    }

    std::string outputPath = oneColorFolder + baseName + " - Average " + std::to_string(average) +
                        " - Tolerance " + std::to_string(start) + ".png";
    Image image1(outputPath.c_str());

    average = !average;

    outputPath = oneColorFolder + baseName + " - Average " + std::to_string(average) +
                        " - Tolerance " + std::to_string(start) + ".png";
    Image image2(outputPath.c_str());

    Image diff = image1;
    diff.diffmap(image2);
    outputPath = "Output/Diffmap/" + baseName + " - Diffmap - Tolerance " + std::to_string(start) + ".png";
    diff.write(outputPath.c_str());
}

// ============================================================================
// PIPELINE PRINCIPAL
// ============================================================================

void processImageTransforms(
    const std::string& inputPath,
    const std::string& baseName,
    const std::vector<int>& thresholdsAndStep,
    int fraction,
    const std::vector<int>& rectanglesToModify,
    const std::vector<int>& tolerance,
    bool totalStepByStepT,
    bool totalBlackAndWhiteT,
    bool totalReversalT,
    bool partialT,
    bool alternatingBlackAndWhite,
    bool oneColor
) {
    const Image image(inputPath.c_str());

    const int first_threshold = thresholdsAndStep[0];
    const int last_threshold = thresholdsAndStep[1];
    const int step = thresholdsAndStep[2];

    // One Color transformation
    if (oneColor) {
        oneColorTransformations(image, baseName, tolerance);
    }

    // Total Step by Step
    if (totalStepByStepT) {
        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_step_by_step_transformations),
            total_step_by_stepoutput_dirs,
            total_step_by_step_suffixes,
            baseName,
            " - ",
            first_threshold,
            last_threshold,
            step,
            true,
            DEFAULT_120_FOLDER,
            {}
        );
    }

    // Total Reversal
    if (totalReversalT) {
        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_reversal_step_by_step_transformations),
            total_step_by_stepoutput_dirs,
            total_step_by_step_suffixes,
            baseName,
            " Reversal - ",
            first_threshold,
            last_threshold,
            step,
            true,
            DEFAULT_120_FOLDER,
            {}
        );
    }

    // Alternating Black and White
    if (alternatingBlackAndWhite) {
        std::vector<int> altParams = {first_threshold, last_threshold, step};
        applyAndSaveGenericTransformations(
            image,
            wrapAlternatingTransforms(total_alternating_black_and_white_transformations),
            total_step_by_stepoutput_dirs,
            total_step_by_step_suffixes,
            baseName,
            " Alternating - ",
            first_threshold,
            last_threshold,
            step,
            true,
            DEFAULT_120_FOLDER,
            altParams
        );
    }

    // Total Black and White
    if (totalBlackAndWhiteT) {
        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_black_and_white_transformations),
            total_black_and_white_output_dirs,
            total_black_and_white_suffixes,
            baseName,
            " - ",
            first_threshold,
            last_threshold,
            step,
            true,
            DEFAULT_120_FOLDER,
            {}
        );
    }

    // Partial Transformations
    if (partialT) {
        std::vector<int> partialParams = {fraction};
        partialParams.insert(partialParams.end(), rectanglesToModify.begin(), rectanglesToModify.end());

        applyAndSaveGenericTransformations(
            image,
            wrapPartialTransforms(partialTransformationsFunc),
            generatePartialOutputDirs(),
            generatePartialSuffixes(),
            baseName,
            " - ",
            first_threshold,
            last_threshold,
            step,
            false,
            "",
            partialParams
        );
    }

}

