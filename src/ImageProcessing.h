#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H
#include "Image.h"
#include "TransformationsConfig.h"

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <cstring>
#include <array>

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
    ImageBuffer(int w, int h, int channels) : buffer(w, h, channels) {}

    void resetFrom(const Image& source) {
        assert(buffer.size == source.size);
        std::memcpy(buffer.data, source.data, buffer.size);
    }



    Image& get() { return buffer; }

    void saveAs(const char* path) {
        buffer.write(path);
    }

    void clear() {
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
        oss << folder120 << baseName << transformationType << suffix
            << " 120";
        /*
        if (includeExtra) {
            oss << " [EXTRA]";
        }
        */
        oss << ".png";
        return oss.str();
    }
};


const std::vector<int> genererRectanglesInDiagonale(int fraction);


void processImageTransforms(
    const std::string& inputFile,
    const std::string& folderName,
    const std::vector<int>& thresholdsAndStep,
    const std::vector<int>& colorNuances,
    int fraction,
    const std::vector<int>& rectanglesToModify,
    const std::vector<int>&  tolerance,
    bool severalColors,
    bool totalBlackAndWhiteT,
    bool totalReversalT,
    bool partialT,
    bool alternatingBlackAndWhite,
    bool oneColor
);



// ============================================================================
// WRAPPERS FOR TYPES OF TRANSFORMATIONS
// ============================================================================

inline std::vector<GenericTransformationFunc> wrapSimpleTransforms(
    const std::vector<TransformationFunc>& transforms
) {
    std::vector<GenericTransformationFunc> wrapped;
    wrapped.reserve(transforms.size());

    for (const auto& transform : transforms) {
        wrapped.emplace_back([transform](Image& img, const int t, const std::vector<int>&) {
            transform(img, t);
        });
    }

    return wrapped;
}


inline std::vector<GenericTransformationFunc> wrapAlternatingTransforms(
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

inline std::vector<GenericTransformationFunc> wrapPartialTransforms(
    const std::vector<PartialTransformationFunc>& transforms
) {
    std::vector<GenericTransformationFunc> wrapped;
    wrapped.reserve(transforms.size());

    for (const auto& transform : transforms) {
        wrapped.emplace_back([transform](Image& img, const int t, const std::vector<int>& params) {
            const int cn = params[0];
            const int fraction = params[1];
            const std::vector<int> rectangles(params.begin() + 2, params.end());
            transform(img, t, cn, fraction, rectangles);
        });
    }

    return wrapped;
}

inline std::vector<GenericTransformationFuncWithColorNuances> wrapPartialTransformsWithRectangles(
    const std::vector<PartialTransformationFunc>& transforms
) {
    std::vector<GenericTransformationFuncWithColorNuances> wrapped;
    wrapped.reserve(transforms.size());

    for (const auto& transform : transforms) {
        wrapped.emplace_back([transform](Image& img, const int i, const int cn, const int fraction,
                const std::vector<int>& rectanglesToModify) {
            transform(img, i, cn, fraction, rectanglesToModify);
        });
    }

    return wrapped;
}


inline std::vector<GenericTransformationFunc>
wrapTwoIntTransforms(const std::vector<TwoIntTransformation>& transforms) {

    std::vector<GenericTransformationFunc> wrapped;
    wrapped.reserve(transforms.size());

    for (const auto& t : transforms) {
        wrapped.emplace_back(
            [t](Image& img, int threshold, const std::vector<int>& params) {
                // params[0] = colorNuance
                t(img, threshold, params[0]);
            }
        );
    }
    return wrapped;
}


#endif
