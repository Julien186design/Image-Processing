#pragma once

// Define THREAD_OFFSET according to your needs
#define THREAD_OFFSET 1  // Example: use all threads minus 1

#include <vector>
#include <string>
#include <functional>
#include <algorithm>

#include "Image.h"

using TransformationFunc = std::function<void(Image&, int)>;
using ReverseTransformationFunc = std::function<void(Image&, int)>;
using AlternatingTransformation = std::function<void(Image&, int, int, int)>;
using PartialTransformationFunc = std::function<void(Image&, int, int, int, const std::vector<int>&)>;
using PartialTransformationFuncByProportion =
    std::function<void(Image&, float, int, int, const std::vector<int>&)>;

using TwoIntTransformationByThreshold = std::function<void(Image&, int, int)>;
using TwoIntTransformationByProportion = std::function<void(Image&, float, int)>;

constexpr const char* OUTPUT_FOLDER = "Output/";
const std::string FOLDER_120 = OUTPUT_FOLDER + std::string("120/");
const std::string FOLDER_VIDEOS = OUTPUT_FOLDER + std::string("Videos/");
const std::string FOLDER_EDGEDETECTOR = OUTPUT_FOLDER + std::string("Edge Detector/");


struct ThresholdParams {
    bool below;
    bool dark;
};

constexpr std::array<ThresholdParams, 4> transformation_params = {{
    {true, true},   // BelowDark
    {true, false},  // BelowLight
    {false, true},  // AboveDark
    {false, false}  // AboveLight
}};


const std::vector<PartialTransformationFuncByProportion> partial_transformations_by_proportion_func = {
    [](Image& img,
        const float proportion, const int cn, const int fraction, const std::vector<int>& rectanglesToModify) {
        img.below_proportion_region_fraction(proportion, cn, fraction, rectanglesToModify, true);
    },
    [](Image& img,
        const float proportion, const int cn, const int fraction, const std::vector<int>& rectanglesToModify) {
        img.below_proportion_region_fraction(proportion, cn, fraction, rectanglesToModify, false);
    },
    [](Image& img,
        const float proportion, const int cn, const int fraction, const std::vector<int>& rectanglesToModify) {
        img.above_proportion_region_fraction(proportion, cn, fraction, rectanglesToModify, true);
    },
    [](Image& img,
        const float proportion, const int cn, const int fraction, const std::vector<int>& rectanglesToModify) {
        img.above_proportion_region_fraction(proportion, cn, fraction, rectanglesToModify, false);
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

inline std::vector<std::string> generatePartialSuffixes() {
    std::vector<std::string> partialSuffixes;
    std::ranges::transform(
        total_step_by_step_suffixes,
        std::back_inserter(partialSuffixes),
        [](const std::string& s) { return s + " Partial"; }
    );
    return partialSuffixes;
}

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

