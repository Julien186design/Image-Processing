#include "ImageProcessing.h"
#include "TransformationsConfig.h"
#include <vector>
#include <functional>
#include <string>
#include <cstring>
#include <omp.h>

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
            // LOGGING
            // printf("  Transformation %zu/%zu: %s\n", i+1, transforms.size(), suffixes[i].c_str());

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

void applyTransformationsWithMultipleColorNuances(
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
    const std::vector<int>& colorNuance
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int currentThreshold = threshold; currentThreshold <= lastThreshold; currentThreshold += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {
            // LOGGING
            // printf("  Transformation %zu/%zu: %s\n", i+1, transforms.size(), suffixes[i].c_str());

            modified.resetFrom(baseImage);

            transforms[i](modified.get(), currentThreshold, colorNuance);

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

    const std::string oneColorFolder = "Output/One Color/";
    const std::string diffmapFolder = "Output/Diffmap/";

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
    outputPath = diffmapFolder + baseName + " - Diffmap - Tolerance " + std::to_string(start) + ".png";
    diff.write(outputPath.c_str());
}

void several_colors_transformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& colorNuances,
    int threshold,
    int lastThreshold,
    int step
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int currentThreshold = threshold; currentThreshold <= lastThreshold; currentThreshold += step) {
        const int threshold3 = 3 * currentThreshold;
        for (size_t transformIdx = 0; transformIdx < colors_nuances_transformations.size(); ++transformIdx) {
            for (int colorNuance : colorNuances) {

                modified.resetFrom(baseImage);

                colors_nuances_transformations[transformIdx](modified.get(), threshold3, colorNuance);

                std::string outputPath = OutputPathBuilder::buildStandard(
                    total_step_by_step_output_dirs[transformIdx],
                    baseName,
                    " - ",
                    total_step_by_step_suffixes[transformIdx],
                    currentThreshold
                );

                // Ajouter colorNuance au nom de fichier
                size_t dotPos = outputPath.rfind(".png");
                if (dotPos != std::string::npos) {
                    outputPath.insert(dotPos, " - CN " + std::to_string(colorNuance));
                }

                modified.saveAs(outputPath.c_str());
            }
        }
    }
}

// ============================================================================
// PIPELINE PRINCIPAL
// ============================================================================

void processImageTransforms(
    const std::string& inputFile,
    const std::string& folderName,
    const std::vector<int>& thresholdsAndStep,
    const std::vector<int>& colorNuances,
    const int fraction,
    const std::vector<int>& rectanglesToModify,
    const std::vector<int>& tolerance,
    const bool severalColors,
    const bool totalBlackAndWhiteT,
    const bool totalReversalT,
    const bool partialT,
    const bool alternatingBlackAndWhite,
    const bool oneColor
) {

    // Extract the base name without the extension
    size_t dotPos = inputFile.find_last_of('.');
    std::string baseName = inputFile.substr(0, dotPos);

    // Load the image from the input path
    std::string inputPath = "Input/" + folderName + "/" + inputFile;
    printf("%s\n", folderName.c_str()); // Print the folder name

    const Image image(inputPath.c_str());

    const int first_threshold = thresholdsAndStep[0];
    const int last_threshold = thresholdsAndStep[1];
    const int step = thresholdsAndStep[2];

    // One Color transformation
    if (oneColor) {
        oneColorTransformations(image, baseName, tolerance);
    }
    /*
    // Total Step by Step
    if (severalColors) {
        printf("DEBUG: Avant applyAndSaveGenericTransformations\n");  // ← AJOUT
        printf("DEBUG: transforms.size() = %zu\n", total_step_by_step_transformations.size());  // ← AJOUT

        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_step_by_step_transformations),
            total_step_by_step_output_dirs,
            total_step_by_step_suffixes,
            baseName,
            " - ",
            first_threshold,
            last_threshold,
            step,
            true,
            DEFAULT_120_FOLDER,
            {},
            twoColors
        );
    }
*/
    if (severalColors) {
        several_colors_transformations(image, baseName, colorNuances, first_threshold, last_threshold, step);
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

    // Total Reversal
    if (totalReversalT) {
        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_reversal_step_by_step_transformations),
            reversal_step_by_step_output_dirs,
            reversal_step_by_step_suffixes,
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

    // Alternating Black and White
    if (alternatingBlackAndWhite) {
        std::vector<int> altParams = {first_threshold, last_threshold, step};
        applyAndSaveGenericTransformations(
            image,
            wrapAlternatingTransforms(total_alternating_black_and_white_transformations),
            total_step_by_step_output_dirs,
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

