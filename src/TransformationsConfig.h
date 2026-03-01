#pragma once

#include "Image.h"
#include "ProcessingConfig.h"

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <array>

// ─── Types ────────────────────────────────────────────────────────────────────

struct ThresholdParams {
    bool below;
    bool dark;
};

using AlternatingTransformation = std::function<void(Image&, int, int, int)>;
using PartialTransformationFuncByProportion =
    std::function<void(Image&, float, int, int, const std::vector<int>&)>;


// ─── Threshold parameter sets ─────────────────────────────────────────────────

constexpr std::array<ThresholdParams, 4> transformation_params = {{
    {true,  true },  // BelowDark
    {true,  false},  // BelowLight
    {false, true },  // AboveDark
    {false, false}   // AboveLight
}};

// ─── Transformation entry: groups a suffix with its output directory ──────────

struct TransformationEntry {
    std::string suffix;
    std::string output_dir;

    // Returns the "partial/square" variant of this entry
    [[nodiscard]] TransformationEntry partial() const {
        std::string dir = output_dir;
        if (!dir.empty() && dir.back() == '/')
            dir.pop_back();
        return { suffix + " Partial", dir + " Square/" };
    }
};

// ─── Step-by-step transformations (BTB, BTW, WTB, WTW) ───────────────────────

const std::vector<TransformationEntry> total_step_by_step_entries = {
    { "BTB", std::string(output_folder) + "BTB/" },
    { "BTW", std::string(output_folder) + "BTW/" },
    { "WTB", std::string(output_folder) + "WTB/" },
    { "WTW", std::string(output_folder) + "WTW/" },
};

// ─── Reversal transformations ─────────────────────────────────────────────────

const std::vector<TransformationEntry> reversal_step_by_step_entries = {
    { "Reversal-BT", std::string(output_folder) + "Reversal-BT/" },
    { "Reversal-WT", std::string(output_folder) + "Reversal-WT/" },
};

// ─── Black-and-white transformations ─────────────────────────────────────────

const std::vector<TransformationEntry> total_black_and_white_entries = {
    { "Black and white - Original", std::string(output_folder) + "Original black and white/" },
    { "Black and white - Reversed", std::string(output_folder) + "Reversed black and white/" },
};

// ─── Helpers to extract suffixes / dirs from an entry list ───────────────────

inline std::vector<std::string> getSuffixes(const std::vector<TransformationEntry>& entries) {
    std::vector<std::string> result;
    result.reserve(entries.size());
    std::ranges::transform(entries, std::back_inserter(result),
        [](const TransformationEntry& e) { return e.suffix; });
    return result;
}

inline std::vector<std::string> getOutputDirs(const std::vector<TransformationEntry>& entries) {
    std::vector<std::string> result;
    result.reserve(entries.size());
    std::ranges::transform(entries, std::back_inserter(result),
        [](const TransformationEntry& e) { return e.output_dir; });
    return result;
}

// ─── Partial variant generation ───────────────────────────────────────────────

inline std::vector<TransformationEntry> generatePartialEntries(
    const std::vector<TransformationEntry>& entries)
{
    std::vector<TransformationEntry> result;
    result.reserve(entries.size());
    std::ranges::transform(entries, std::back_inserter(result),
        [](const TransformationEntry& e) { return e.partial(); });
    return result;
}
