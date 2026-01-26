#pragma once

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cstdint>

#include "Image.h"

using TransformationFunc = std::function<void(Image&, int)>;
using ReverseTransformationFunc = std::function<void(Image&, int)>;
using AlternatingTransformation = std::function<void(Image&, int, int, int)>;
using PartialTransformationFunc = std::function<void(Image&, int, int, int, const std::vector<int>&)>;

using TwoIntTransformationByThreshold = std::function<void(Image&, int, int)>;
using TwoIntTransformationByProportion = std::function<void(Image&, float, int)>;
using TwoIntTransformation_AVX2 = std::function<void(Image&, int, std::uint8_t)>;
//Advanced Vector Extensions 2

constexpr const char* OUTPUT_FOLDER = "Output/";
const std::string FOLDER_120 = OUTPUT_FOLDER + std::string("120/");
const std::string FOLDER_VIDEOS = OUTPUT_FOLDER + std::string("Videos/");
const std::string FOLDER_EDGEDETECTOR = OUTPUT_FOLDER + std::string("Edge Detector/");

const std::vector<TwoIntTransformationByThreshold> transformations_by_threshold = {
    [](Image& img, const int i, const int cn){img.below_threshold(i, cn, true); },
    [](Image& img, const int i, const int cn){img.below_threshold(i, cn, false); },
    [](Image& img, const int i, const int cn){img.aboveThreshold(i, cn, true); },
    [](Image& img, const int i, const int cn){img.aboveThreshold(i, cn, false); }
};

const std::vector<TwoIntTransformationByProportion> transformations_by_proportion = {
    [](Image& img, const float prop, const int cn){img.belowProportion(prop, cn, true); },
    [](Image& img, const float prop, const int cn){img.belowProportion(prop, cn, false); },
    [](Image& img, const float prop, const int cn){img.aboveProportion(prop, cn, true); },
    [](Image& img, const float prop, const int cn){img.aboveProportion(prop, cn, false); }
};

// Define the type for transformations that take an Image&, an int, and a uint8_t


// Declare the vector of transformations
const std::vector<TwoIntTransformation_AVX2> colors_nuances_transformations_AVX2 = {
    [](Image& img, const int threshold, const std::uint8_t colorNuance) {
        img.darkenBelowThreshold_ColorNuance_AVX2(threshold, colorNuance);
    },
    [](Image& img, const int threshold, const std::uint8_t colorNuance) {
        img.whitenBelowThreshold_ColorNuance_AVX2(threshold, colorNuance);
    },
    [](Image& img, const int threshold, const std::uint8_t colorNuance) {
        img.darkenAboveThreshold_ColorNuance_AVX2(threshold, colorNuance);
    },
    [](Image& img, const int threshold, const std::uint8_t colorNuance) {
        img.whitenAboveThreshold_ColorNuance_AVX2(threshold, colorNuance);
    }
};


const std::vector<PartialTransformationFunc> partialTransformationsFunc = {
    [](Image& img, const int i, const int cn, const int fraction, const std::vector<int>& rectanglesToModify) {
        img.darkenBelowThresholdRegionFraction(i, cn, fraction, rectanglesToModify);
    },
    [](Image& img, const int i, const int cn, const int fraction, const std::vector<int>& rectanglesToModify) {
        img.whitenBelowThresholdRegionFraction(i, cn, fraction, rectanglesToModify);
    },
    [](Image& img, const int i, const int cn, const int fraction, const std::vector<int>& rectanglesToModify) {
        img.darkenAboveThresholdRegionFraction(i, cn, fraction, rectanglesToModify);
    },
    [](Image& img, const int i, const int cn, const int fraction, const std::vector<int>& rectanglesToModify) {
        img.whitenAboveThresholdRegionFraction(i, cn, fraction, rectanglesToModify);
    }
};

const std::vector<ReverseTransformationFunc> total_reversal_step_by_step_transformations = {
    [](Image& img, const int i) { img.reverseAboveThreshold(i); },
    [](Image& img, const int i) { img.reverseBelowThreshold(i); }
};

const std::vector<AlternatingTransformation> total_alternating_black_and_white_transformations = {
    [](Image& img, const int i, const int first_t, const int last_t) {
        img.alternatelyDarkenAndWhitenBelowTheThreshold(i, first_t, last_t);
    },
    [](Image& img, const int i, const int first_t, const int last_t) {
        img.alternatelyDarkenAndWhitenAboveTheThreshold(i, first_t, last_t);
    }
};

const std::vector<TransformationFunc> total_black_and_white_transformations = {
    [](Image& img, const int i) { img.original_black_and_white(i); },
    [](Image& img, const int i) { img.reversed_black_and_white(i); }
};

const std::vector<std::string> total_step_by_step_suffixes = {
    "BTB", "BTW", "WTB", "WTW"
};

const std::vector<std::string> reversal_step_by_step_suffixes = {
    "Reversal-BT", "Reversal-WT"
};

const std::vector<std::string> total_black_and_white_suffixes = {
    "Black and white - Original", "Black and white - Reversed"
};


inline std::vector<std::string> generatePartialSuffixes() {
    std::vector<std::string> partialSuffixes;
    std::ranges::transform(
        total_step_by_step_suffixes,
        std::back_inserter(partialSuffixes),
        [](const std::string& s) { return s + " Partial"; }
    );
    return partialSuffixes;
}


const std::vector total_step_by_step_output_dirs = {
    std::string(OUTPUT_FOLDER) + "BTB/", std::string(OUTPUT_FOLDER) + "BTW/",
    std::string(OUTPUT_FOLDER) + "WTB/", std::string(OUTPUT_FOLDER) + "WTW/"
};

const std::vector reversal_step_by_step_output_dirs = {
    std::string(OUTPUT_FOLDER) + "Reversal-BT/", std::string(OUTPUT_FOLDER) + "Reversal-WT/"
};

const std::vector total_black_and_white_output_dirs = {
    std::string(OUTPUT_FOLDER) + "Original black and white/",
    std::string(OUTPUT_FOLDER) + "Reversed black and white/"
};


inline std::vector<std::string> generatePartialOutputDirs() {
    std::vector<std::string> partialOutputDirs;
    std::ranges::transform(
        total_step_by_step_output_dirs,
        std::back_inserter(partialOutputDirs),
        [](const std::string& s) {
            if (s.empty()) return s;
            return s.substr(0, s.size() - 1) + " Square/";
        }
    );
    return partialOutputDirs;
}

