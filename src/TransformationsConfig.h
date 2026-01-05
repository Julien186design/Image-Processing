#pragma once

#include "ImageProcessing.h"
#include <vector>
#include <string>
#include <functional>
#include <algorithm>


using TransformationFunc = std::function<void(Image&, int)>;
using ReverseTransformationFunc = std::function<void(Image&, int)>;
using AlternatingTransformation = std::function<void(Image&, int, int, int)>;
using PartialTransformationFunc = std::function<void(Image&, int, int, const std::vector<int>&)>;

constexpr const char* DEFAULT_120_FOLDER = "Output/120/";

const std::vector<TransformationFunc> total_step_by_step_transformations = {
    [](Image& img, int i) { img.darkenBelowThreshold(i); },
    [](Image& img, int i) { img.whitenBelowThreshold(i); },
    [](Image& img, int i) { img.darkenAboveThreshold(i); },
    [](Image& img, int i) { img.whitenAboveThreshold(i); }
};

const std::vector<ReverseTransformationFunc> total_reversal_step_by_step_transformations = {
    [](Image& img, int i) { img.reverseAboveThreshold(i); },
    [](Image& img, int i) { img.reverseBelowThreshold(i); }
};

const std::vector<AlternatingTransformation> total_alternating_black_and_white_transformations = {
    [](Image& img, int i, int first_t, int last_t) { img.alternatelyDarkenAndWhitenBelowTheThreshold(i, first_t, last_t); },
    [](Image& img, int i, int first_t, int last_t) { img.alternatelyDarkenAndWhitenAboveTheThreshold(i, first_t, last_t); }
};

const std::vector<TransformationFunc> total_black_and_white_transformations = {
    [](Image& img, int i) { img.original_black_and_white(i); },
    [](Image& img, int i) { img.reversed_black_and_white(i); }
};

const std::vector<PartialTransformationFunc> partialTransformationsFunc = {
    [](Image& img, int i, int fraction, const std::vector<int>& rectanglesToModify) {
        img.darkenBelowThresholdRegionFraction(i, fraction, rectanglesToModify);
    },
    [](Image& img, int i, int fraction, const std::vector<int>& rectanglesToModify) {
        img.whitenBelowThresholdRegionFraction(i, fraction, rectanglesToModify);
    },
    [](Image& img, int i, int fraction, const std::vector<int>& rectanglesToModify) {
        img.darkenAboveThresholdRegionFraction(i, fraction, rectanglesToModify);
    },
    [](Image& img, int i, int fraction, const std::vector<int>& rectanglesToModify) {
        img.whitenAboveThresholdRegionFraction(i, fraction, rectanglesToModify);
    }
};


const std::vector<std::string> total_step_by_step_suffixes = {
    "BTB", "BTW", "WTB", "WTW"
};

const std::vector<std::string> total_black_and_white_suffixes = {
    "Original black and white", "Reversed black and white"
};


inline std::vector<std::string> generatePartialSuffixes() {
    std::vector<std::string> partialSuffixes;
    std::transform(
        total_step_by_step_suffixes.begin(), total_step_by_step_suffixes.end(),
        std::back_inserter(partialSuffixes),
        [](const std::string& s) { return s + " Partial"; }
    );
    return partialSuffixes;
}

// Vecteurs de répertoires de sortie
const std::vector<std::string> total_step_by_stepoutput_dirs = {
    "Output/BTB/", "Output/BTW/", "Output/WTB/", "Output/WTW/"
};

const std::vector<std::string> total_black_and_white_output_dirs = {
    "Output/Original black and white/", "Output/Reversed black and white/"
};

// Fonction pour générer les répertoires de sortie partiels
inline std::vector<std::string> generatePartialOutputDirs() {
    std::vector<std::string> partialOutputDirs;
    std::transform(
        total_step_by_stepoutput_dirs.begin(), total_step_by_stepoutput_dirs.end(),
        std::back_inserter(partialOutputDirs),
        [](const std::string& s) {
            if (s.empty()) return s;
            return s.substr(0, s.size() - 1) + " Square/";
        }
    );
    return partialOutputDirs;
}
