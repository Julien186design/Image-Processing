#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define BYTE_BOUND(value) value < 0 ? 0 : (value > 255 ? 255 : value)

/*
 *CLion on Linux Mint
 *Run/Edit Configurations
 *Working directory : $PROJECT8DIR$ | /[...]/CLion Projects/Image Processing

*/
#include <stb_image.h>
#include <stb_image_write.h>
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <fmt/color.h>
#include <cmath>     // for std::round
#include <ranges>

#include "Image.h"

EdgeDetectorResult process_edge_detection_core(
    const uint8_t* grayData,
    const int width,
    const int height,
    const double threshold
) {
    const size_t imgSize = width * height;

    // Pre-computed Gaussian kernel
    constexpr double inv16 = 1.0 / 16.0;
    constexpr double gauss[9] = {
        inv16, 2*inv16, inv16,
        2*inv16, 4*inv16, 2*inv16,
        inv16, 2*inv16, inv16
    };

    std::vector<double> blurData(imgSize, 0.0);
    std::vector<double> tx(imgSize), ty(imgSize);
    std::vector<double> gx(imgSize), gy(imgSize);
    std::vector<double> g(imgSize), theta(imgSize);

    // === Gaussian blur ===
    #pragma omp parallel for
    for (int r = 1; r < height - 1; ++r) {
        for (int c = 1; c < width - 1; ++c) {
            double sum = 0.0;
            for (int kr = -1; kr <= 1; ++kr) {
                for (int kc = -1; kc <= 1; ++kc) {
                    sum += grayData[(r + kr) * width + (c + kc)] *
                           gauss[(kr + 1) * 3 + (kc + 1)];
                }
            }
            blurData[r * width + c] = sum;
        }
    }

    // === Scharr separable convolution ===
    #pragma omp parallel for
    for (int r = 0; r < height; ++r) {
        for (int c = 1; c < width - 1; ++c) {
            const size_t idx = r * width + c;
            tx[idx] = blurData[idx + 1] - blurData[idx - 1];
            ty[idx] = 47 * blurData[idx + 1] + 162 * blurData[idx] + 47 * blurData[idx - 1];
        }
    }

    #pragma omp parallel for
    for (int c = 1; c < width - 1; ++c) {
        for (int r = 1; r < height - 1; ++r) {
            const size_t idx = r * width + c;
            gx[idx] = 47 * tx[idx + width] + 162 * tx[idx] + 47 * tx[idx - width];
            gy[idx] = ty[idx + width] - ty[idx - width];
        }
    }

    // === Magnitude/angle + min/max reduction ===
    double mx = -INFINITY, mn = INFINITY;
    #pragma omp parallel
    {
        double local_mx = -INFINITY, local_mn = INFINITY;

        #pragma omp for nowait
        for (int k = 0; k < imgSize; ++k) {
            g[k] = std::sqrt(gx[k] * gx[k] + gy[k] * gy[k]);
            theta[k] = std::atan2(gy[k], gx[k]);
            local_mx = std::max(local_mx, g[k]);
            local_mn = std::min(local_mn, g[k]);
        }

        #pragma omp critical
        {
            mx = std::max(mx, local_mx);
            mn = std::min(mn, local_mn);
        }
    }

    // === HSL→RGB conversion ===
    std::vector<uint8_t> outputRGB(imgSize * 3);
    const double range = (mx == mn) ? 1.0 : (mx - mn);

    #pragma omp parallel for
    for (int k = 0; k < imgSize; ++k) {
        const double h = theta[k] * 180.0 / M_PI + 180.0;
        const double v = ((g[k] - mn) / range > threshold) ? (g[k] - mn) / range : 0.0;
        const double s = v, l = v;

        const double c = (1 - std::abs(2 * l - 1)) * s;
        const double x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
        const double m = l - c / 2.0;

        double rt = 0, gt = 0, bt = 0;
        if (h < 60)       { rt = c; gt = x; }
        else if (h < 120) { rt = x; gt = c; }
        else if (h < 180) { gt = c; bt = x; }
        else if (h < 240) { gt = x; bt = c; }
        else if (h < 300) { bt = c; rt = x; }
        else              { bt = x; rt = c; }

        outputRGB[k * 3]     = static_cast<uint8_t>(255 * (rt + m));
        outputRGB[k * 3 + 1] = static_cast<uint8_t>(255 * (gt + m));
        outputRGB[k * 3 + 2] = static_cast<uint8_t>(255 * (bt + m));
    }

    return {std::move(outputRGB), mn, mx};
}


Image::Image(const char* filename, const int channel_force) {
	if(read(filename, channel_force)) {
		printf("Read %s\n", filename);
		size = w*h*channels;
	} else {
		printf("Failed to read %s\n", filename);
	}
}

Image::Image(const int w, const int h, const int channels)
	: w(w), h(h), channels(channels) {
	size = w * h * channels;
	data = new uint8_t[size];
	memset(data, 0, size);
}

Image::Image(const Image& img) : Image(img.w, img.h, img.channels) {
	memcpy(data, img.data, size);
}

Image::~Image() {
	// stbi_image_free(data);
	delete[] data;

}

bool Image::read(const char* filename, const int channel_force) {
	data = stbi_load(filename, &w, &h, &channels, channel_force);
	channels = channel_force == 0 ? channels : channel_force;
	return data != nullptr; // used to be NULL
}

bool Image::write(const char* filename) {
	ImageType type = get_file_type(filename);
	int success;
	switch (type) {
		case PNG:
			success = stbi_write_png(filename, w, h, channels, data, w*channels);
			break;
		case BMP:
			success = stbi_write_bmp(filename, w, h, channels, data);
			break;
		case JPG:
			success = stbi_write_jpg(filename, w, h, channels, data, 100);
			break;
		case TGA:
			success = stbi_write_tga(filename, w, h, channels, data);
			break;
	}
	if (success != 0) {
		printf("\e[32mWrote \e[36m%s\e[0m, %d, %d, %d, %zu\n", filename, w, h, channels, size);
		return true;
	} else {
		printf("\e[31;1m Failed to write \e[36m%s\e[0m, %d, %d, %d, %zu\n", filename, w, h, channels, size); //width, height
		return false;
	}
}

ImageType Image::get_file_type(const char* filename) {
	const char* ext = strrchr(filename, '.');
	if(ext != nullptr) {
		if(strcmp(ext, ".png") == 0) {
			return PNG;
		} else if(strcmp(ext, ".jpg") == 0) {
			return JPG;
		} else if(strcmp(ext, ".bmp") == 0) {
			return BMP;
		} else if(strcmp(ext, ".tga") == 0) {
			return TGA;
		}
	}
	return PNG;
}



Image& Image::below_threshold(const int threshold, const int cn, const bool useDarkNuance) {
	const int newColor = useDarkNuance ? cn : 255 - cn;
	for(size_t i = 0; i < size; i += static_cast<size_t>(channels)) {
		if ((data[i] + data[i + 1] + data[i + 2]) < threshold) {
			data[i] = data[i + 1] = data[i + 2] = newColor;
		}
	}

	return *this;
}

Image& Image::aboveThreshold(const int threshold, const int cn, const bool useDarkNuance) {
	const int newColor = useDarkNuance ? cn : 255 - cn;
	for(size_t i = 0; i < size; i += static_cast<size_t>(channels)) {
		if ((data[i] + data[i + 1] + data[i + 2]) > threshold) {
			data[i] = data[i + 1] = data[i + 2] = newColor;
		}
	}

	return *this;
}

Image& Image::belowProportion(const float proportion, const int cn, const bool useDarkNuance) {
    if (proportion <= 0.0f || proportion >= 1.0f) return *this;

    // Calculer les valeurs RGB de tous les pixels
    const size_t pixelCount = size / static_cast<size_t>(channels);
    std::vector<int> rgbValues(pixelCount);

    for(size_t i = 0, idx = 0; i < size; i += static_cast<size_t>(channels), ++idx) {
        rgbValues[idx] = data[i] + data[i + 1] + data[i + 2];
    }

    // Trier pour trouver le seuil correspondant à la proportion
    std::vector<int> sortedValues = rgbValues;
	std::ranges::sort(sortedValues);
    const int threshold = sortedValues[static_cast<size_t>(pixelCount * proportion)];

    // Appliquer la transformation
    const int newColor = useDarkNuance ? cn : 255 - cn;
    for(size_t i = 0; i < size; i += static_cast<size_t>(channels)) {
        const int rgb = data[i] + data[i + 1] + data[i + 2];
        if (rgb <= threshold) {
            data[i] = data[i + 1] = data[i + 2] = newColor;
        }
    }
    return *this;
}

Image& Image::aboveProportion(const float proportion, const int cn, const bool useDarkNuance) {
    if (proportion <= 0.0f || proportion >= 1.0f) return *this;

    // Calculer les valeurs RGB de tous les pixels
    const size_t pixelCount = size / static_cast<size_t>(channels);
    std::vector<int> rgbValues(pixelCount);

    for(size_t i = 0, idx = 0; i < size; i += static_cast<size_t>(channels), ++idx) {
        rgbValues[idx] = data[i] + data[i + 1] + data[i + 2];
    }

    // Trier pour trouver le seuil correspondant à la proportion
    std::vector<int> sortedValues = rgbValues;
	std::ranges::sort(sortedValues, std::greater{});
    const int threshold = sortedValues[static_cast<size_t>(pixelCount * proportion)];

    // Appliquer la transformation
    const int newColor = useDarkNuance ? cn : 255 - cn;
    for(size_t i = 0; i < size; i += static_cast<size_t>(channels)) {
        const int rgb = data[i] + data[i + 1] + data[i + 2];
        if (rgb >= threshold) {
            data[i] = data[i + 1] = data[i + 2] = newColor;
        }
    }
    return *this;
}

void Image::simplify_pixel(
	uint8_t& r, uint8_t& g, uint8_t& b,
	const uint8_t r_val_third, const uint8_t g_val_third, const uint8_t b_val_third,
	const uint8_t r_val_half, const uint8_t g_val_half, const uint8_t b_val_half,
	const uint8_t r_val_full, const uint8_t g_val_full, const uint8_t b_val_full,
	const int tolerance
) {
	const bool r_eq_g = approx_equal(r, g, tolerance);
	const bool r_eq_b = approx_equal(r, b, tolerance);
	const bool g_eq_b = approx_equal(g, b, tolerance);

	// Cas 1 : Tous les canaux sont égaux
	if (r_eq_g && r_eq_b && g_eq_b) {
		r = r_val_third;
		g = g_val_third;
		b = b_val_third;
		return;
	}

	// Cas 2 : Deux canaux sont égaux
	if (r_eq_g) {
		r = r_val_half;
		g = g_val_half;
		b = 0;
		return;
	} else if (r_eq_b) {
		r = r_val_half;
		g = 0;
		b = b_val_half;
		return;
	} else if (g_eq_b) {
		r = 0;
		g = g_val_half;
		b = b_val_half;
		return;
	}

	// Cas 3 : Tous les canaux sont distincts
	const ChannelIndices indices = get_channel_indices(r, g, b);
	const uint8_t avgs_full[] = {r_val_full, g_val_full, b_val_full};
	const uint8_t avgs_half[] = {r_val_half, g_val_half, b_val_half};

	uint8_t new_vals[3] = {0, 0, 0};
	new_vals[indices.max] = avgs_full[indices.max];
	new_vals[indices.mid] = avgs_half[indices.mid];
	new_vals[indices.min] = 0;

	r = new_vals[0];
	g = new_vals[1];
	b = new_vals[2];
}

Image& Image::simplify_to_dominant_color_combinations_with_average(const int tolerance) {
	for (size_t i = 0; i < size; i += channels) {
		uint8_t& r = data[i];
		uint8_t& g = data[i + 1];
		uint8_t& b = data[i + 2];

		const uint8_t r_third = avg_u8_round(r, SimpleColors::ONE_THIRD);
		const uint8_t g_third = avg_u8_round(g, SimpleColors::ONE_THIRD);
		const uint8_t b_third = avg_u8_round(b, SimpleColors::ONE_THIRD);

		const uint8_t r_half = avg_u8_round(r, SimpleColors::HALF);
		const uint8_t g_half = avg_u8_round(g, SimpleColors::HALF);
		const uint8_t b_half = avg_u8_round(b, SimpleColors::HALF);

		const uint8_t r_full = avg_u8_round(r, SimpleColors::FULL);
		const uint8_t g_full = avg_u8_round(g, SimpleColors::FULL);
		const uint8_t b_full = avg_u8_round(b, SimpleColors::FULL);

		simplify_pixel(
			r, g, b,
			r_third, g_third, b_third,
			r_half, g_half, b_half,
			r_full, g_full, b_full,
			tolerance
		);
	}
	return *this;
}

Image& Image::simplify_to_dominant_color_combinations_without_average(
			const int tolerance,
			const uint8_t r_third, const uint8_t g_third, const uint8_t b_third,
			const uint8_t r_half, const uint8_t g_half, const uint8_t b_half,
			const uint8_t r_full, const uint8_t g_full, const uint8_t b_full
		) {

	for (size_t i = 0; i < size; i += channels) {
		uint8_t& r = data[i];
		uint8_t& g = data[i + 1];
		uint8_t& b = data[i + 2];

		simplify_pixel(
			r, g, b,
			r_third, g_third, b_third,
			r_half, g_half, b_half,
			r_full, g_full, b_full,
			tolerance
		);
	}
	return *this;
}

Image& Image::reverseAboveThreshold(const int threshold) {
	for(size_t i = 0; i < size; i += channels) {
		const int rgb = (data[i] + data[i + 1] + data[i + 2]);
		if (rgb < threshold) {
			data[i] = 255 - data[i];
			data[i + 1] = 255 - data[i + 1];
			data[i + 2] = 255 - data[i + 2];
		}
	}
	return *this;
}

Image& Image::reverseBelowThreshold(const int threshold) {
	for(size_t i = 0; i < size; i+=channels) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb > threshold) {
			data[i] = 255 - data[i];
			data[i + 1] = 255 - data[i + 1];
			data[i + 2] = 255 - data[i + 2];
		}
	}
	return *this;
}

Image& Image::original_black_and_white(const int threshold) {
	for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb > threshold) {
			data[i] = data[i + 1] = data[i + 2] = 255;
		} else {
			data[i] = data[i + 1] = data[i + 2] = 0;
		}
	}
	return *this;
}

Image& Image::reversed_black_and_white(const int threshold) {
	for (size_t i = 0; i < size; i += channels) {

		const int rgb = data[i] + data[i + 1] + data[i + 2];

		const uint8_t value = (rgb < threshold) ? 255 : 0;

		data[i] = data[i + 1] = data[i + 2] = value;
	}
	return *this;
}

// in construction
Image& Image::alternatelyDarkenAndWhitenBelowTheThreshold(int s, int first_threshold,	int last_threshold) {
	const int threshold3 = 3 * s;
	for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb < threshold3) {
			data[i] = data[i + 1] = data[i + 2] = 0;
		}
	}
	return *this;
}

Image& Image::alternatelyDarkenAndWhitenAboveTheThreshold(int s, int first_threshold,	int last_threshold) {
	const int threshold3 = 3 * s;
	for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb > threshold3) {
			data[i] = data[i + 1] = data[i + 2] = 0;
		}
	}
	return *this;
}


//fraction by rectangles

template<typename ConditionFunc, typename TransformFunc>
Image& Image::applyThresholdTransformationRegionFraction(
	int threshold,
	const int fraction,
	const std::vector<int>& rectanglesToModify,
	ConditionFunc condition,
	TransformFunc transformation
) {
	if (rectanglesToModify.empty()) return *this;

	const int numRectanglesPerRow = 1 << fraction;
	const int totalRectangles = numRectanglesPerRow * numRectanglesPerRow;
	const int rectWidth = w / numRectanglesPerRow;
	const int rectHeight = h / numRectanglesPerRow;
	const uint8_t transformValue = transformation();

	for (const int rectIndex : rectanglesToModify) {
		if (rectIndex < 0 || rectIndex >= totalRectangles) continue;

		const int rectRow = rectIndex / numRectanglesPerRow;
		const int rectCol = rectIndex % numRectanglesPerRow;
		const int startX = rectCol * rectWidth;
		const int startY = rectRow * rectHeight;
		const int endX = std::min((rectCol + 1) * rectWidth, w);
		const int endY = std::min((rectRow + 1) * rectHeight, h);

		for (int y = startY; y < endY; ++y) {
			const size_t rowOffset = y * w * channels;
			for (int x = startX; x < endX; ++x) {
				const size_t i = rowOffset + x * channels;
				const int rgb = data[i] + data[i + 1] + data[i + 2];

				if (condition(rgb)) {
					data[i] = data[i + 1] = data[i + 2] = transformValue;
				}
			}
		}
	}
	return *this;
}

Image& Image::darkenBelowThresholdRegionFraction(
	int threshold,
	int cn,
	const int fraction,
	const std::vector<int>& rectanglesToModify) {

	return applyThresholdTransformationRegionFraction(
	   threshold,
	   fraction,
	   rectanglesToModify,
	   [threshold](const int rgb) { return rgb < threshold; },
	   [cn]() -> uint8_t { return cn; }
	);
}

Image& Image::whitenBelowThresholdRegionFraction(
	int threshold,
	const int cn,
	const int fraction,
	const std::vector<int>& rectanglesToModify) {

	const int newColor = 255 - cn;

		return applyThresholdTransformationRegionFraction(
		   threshold,
		   fraction,
		   rectanglesToModify,
		   [threshold](const int rgb) { return rgb < threshold; },
		   [newColor]() -> uint8_t { return newColor; }
		);
	}

Image& Image::darkenAboveThresholdRegionFraction(
	int threshold,
	const int cn,
	const int fraction,
	const std::vector<int>& rectanglesToModify) {

	return applyThresholdTransformationRegionFraction(
	   threshold,
	   fraction,
	   rectanglesToModify,
	   [threshold](const int rgb) { return rgb > threshold; },
	   [cn]() -> uint8_t { return cn; }
	);
}

Image& Image::whitenAboveThresholdRegionFraction(
	int threshold,
	const int cn,
	const int fraction,
	const std::vector<int>& rectanglesToModify) {

	const int newColor = 255 - cn;

		return applyThresholdTransformationRegionFraction(
		   threshold,
		   fraction,
		   rectanglesToModify,
		   [threshold](const int rgb) { return rgb > threshold; },
		   [newColor]() -> uint8_t { return newColor; }
		);
	}

Image& Image::darkenBelowThreshold_ColorNuance_AVX2(const int threshold, const std::uint8_t cn)
{
    // Set the constant color value for darkening
    const __m256i v_cn = _mm256_set1_epi8(static_cast<char>(cn));
    // Set the threshold value for comparison
	const __m256i v_th = _mm256_set1_epi8(static_cast<char>(threshold));

    std::uint8_t* p = data;
    std::uint8_t* end = data + size - 96; // 32 RGB pixels (96 bytes)

    // Process 32 pixels at a time using AVX2
    for (; p <= end; p += 96)
    {
        // Load 96 bytes (32 RGB pixels) into three 256-bit registers
    	__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    	__m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    	__m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64));

        // Extract R, G, B components for the first 32 bytes (a)
        __m256i r_a = _mm256_and_si256(a, _mm256_set1_epi32(0xFF));
        __m256i g_a = _mm256_and_si256(_mm256_srli_epi32(a, 8), _mm256_set1_epi32(0xFF));
        __m256i b_a = _mm256_and_si256(_mm256_srli_epi32(a, 16), _mm256_set1_epi32(0xFF));

        // Extract R, G, B components for the next 32 bytes (b)
        __m256i r_b = _mm256_and_si256(b, _mm256_set1_epi32(0xFF));
        __m256i g_b = _mm256_and_si256(_mm256_srli_epi32(b, 8), _mm256_set1_epi32(0xFF));
        __m256i b_b = _mm256_and_si256(_mm256_srli_epi32(b, 16), _mm256_set1_epi32(0xFF));

        // Extract R, G, B components for the last 32 bytes (c)
        __m256i r_c = _mm256_and_si256(c, _mm256_set1_epi32(0xFF));
        __m256i g_c = _mm256_and_si256(_mm256_srli_epi32(c, 8), _mm256_set1_epi32(0xFF));
        __m256i b_c = _mm256_and_si256(_mm256_srli_epi32(c, 16), _mm256_set1_epi32(0xFF));

        // Convert R, G, B components to 16-bit for the first 16 pixels (lower 128 bits)
        __m256i r16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_a));
        __m256i g16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_a));
        __m256i b16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_a));

        __m256i r16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_b));
        __m256i g16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_b));
        __m256i b16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_b));

        __m256i r16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_c));
        __m256i g16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_c));
        __m256i b16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_c));

        // Convert R, G, B components to 16-bit for the next 16 pixels (upper 128 bits)
        __m256i r16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_a, 1));
        __m256i g16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_a, 1));
        __m256i b16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_a, 1));

        __m256i r16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_b, 1));
        __m256i g16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_b, 1));
        __m256i b16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_b, 1));

        __m256i r16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_c, 1));
        __m256i g16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_c, 1));
        __m256i b16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_c, 1));

    	// Calculate the sum of R + G + B for each group of 16 pixels (unsigned saturated)
    	__m256i sum_a = _mm256_adds_epu16(_mm256_adds_epu16(r16_a, g16_a), b16_a);
    	__m256i sum_a_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_a_high, g16_a_high), b16_a_high);

    	__m256i sum_b = _mm256_adds_epu16(_mm256_adds_epu16(r16_b, g16_b), b16_b);
    	__m256i sum_b_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_b_high, g16_b_high), b16_b_high);

    	__m256i sum_c = _mm256_adds_epu16(_mm256_adds_epu16(r16_c, g16_c), b16_c);
    	__m256i sum_c_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_c_high, g16_c_high), b16_c_high);


        // Compare the sum to the threshold
        __m256i mask_a = _mm256_cmpgt_epi16(v_th, sum_a);
        __m256i mask_a_high = _mm256_cmpgt_epi16(v_th, sum_a_high);

        __m256i mask_b = _mm256_cmpgt_epi16(v_th, sum_b);
        __m256i mask_b_high = _mm256_cmpgt_epi16(v_th, sum_b_high);

        __m256i mask_c = _mm256_cmpgt_epi16(v_th, sum_c);
        __m256i mask_c_high = _mm256_cmpgt_epi16(v_th, sum_c_high);

        // Combine masks for each register
        __m256i mask8_a = _mm256_packus_epi16(mask_a, mask_a_high);
        __m256i mask8_b = _mm256_packus_epi16(mask_b, mask_b_high);
        __m256i mask8_c = _mm256_packus_epi16(mask_c, mask_c_high);

        // Apply the mask to darken pixels
        a = _mm256_blendv_epi8(a, v_cn, mask8_a);
        b = _mm256_blendv_epi8(b, v_cn, mask8_b);
        c = _mm256_blendv_epi8(c, v_cn, mask8_c);

        // Store the results back to memory
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a);
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 32), b);
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 64), c);

    }

    // Process remaining pixels with a scalar loop
    for (; p < data + size; p += 3)
    {
        if (p[0] + p[1] + p[2] < threshold)
        {
            p[0] = p[1] = p[2] = cn;
        }
    }

    return *this;
}

Image& Image::whitenBelowThreshold_ColorNuance_AVX2(const int threshold, const std::uint8_t cn)
{
    // Set the constant color value for darkening
    const __m256i v_cn = _mm256_set1_epi8(static_cast<char>(255 - cn));
    // Set the threshold value for comparison
	const __m256i v_th = _mm256_set1_epi8(static_cast<char>(threshold));

    std::uint8_t* p = data;
    std::uint8_t* end = data + size - 96; // 32 RGB pixels (96 bytes)

    // Process 32 pixels at a time using AVX2
    for (; p <= end; p += 96)
    {
        // Load 96 bytes (32 RGB pixels) into three 256-bit registers
    	__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    	__m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    	__m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64));

        // Extract R, G, B components for the first 32 bytes (a)
        __m256i r_a = _mm256_and_si256(a, _mm256_set1_epi32(0xFF));
        __m256i g_a = _mm256_and_si256(_mm256_srli_epi32(a, 8), _mm256_set1_epi32(0xFF));
        __m256i b_a = _mm256_and_si256(_mm256_srli_epi32(a, 16), _mm256_set1_epi32(0xFF));

        // Extract R, G, B components for the next 32 bytes (b)
        __m256i r_b = _mm256_and_si256(b, _mm256_set1_epi32(0xFF));
        __m256i g_b = _mm256_and_si256(_mm256_srli_epi32(b, 8), _mm256_set1_epi32(0xFF));
        __m256i b_b = _mm256_and_si256(_mm256_srli_epi32(b, 16), _mm256_set1_epi32(0xFF));

        // Extract R, G, B components for the last 32 bytes (c)
        __m256i r_c = _mm256_and_si256(c, _mm256_set1_epi32(0xFF));
        __m256i g_c = _mm256_and_si256(_mm256_srli_epi32(c, 8), _mm256_set1_epi32(0xFF));
        __m256i b_c = _mm256_and_si256(_mm256_srli_epi32(c, 16), _mm256_set1_epi32(0xFF));

        // Convert R, G, B components to 16-bit for the first 16 pixels (lower 128 bits)
        __m256i r16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_a));
        __m256i g16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_a));
        __m256i b16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_a));

        __m256i r16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_b));
        __m256i g16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_b));
        __m256i b16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_b));

        __m256i r16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_c));
        __m256i g16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_c));
        __m256i b16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_c));

        // Convert R, G, B components to 16-bit for the next 16 pixels (upper 128 bits)
        __m256i r16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_a, 1));
        __m256i g16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_a, 1));
        __m256i b16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_a, 1));

        __m256i r16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_b, 1));
        __m256i g16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_b, 1));
        __m256i b16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_b, 1));

        __m256i r16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_c, 1));
        __m256i g16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_c, 1));
        __m256i b16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_c, 1));

    	// Calculate the sum of R + G + B for each group of 16 pixels (unsigned saturated)
    	__m256i sum_a = _mm256_adds_epu16(_mm256_adds_epu16(r16_a, g16_a), b16_a);
    	__m256i sum_a_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_a_high, g16_a_high), b16_a_high);

    	__m256i sum_b = _mm256_adds_epu16(_mm256_adds_epu16(r16_b, g16_b), b16_b);
    	__m256i sum_b_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_b_high, g16_b_high), b16_b_high);

    	__m256i sum_c = _mm256_adds_epu16(_mm256_adds_epu16(r16_c, g16_c), b16_c);
    	__m256i sum_c_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_c_high, g16_c_high), b16_c_high);


        // Compare the sum to the threshold
        __m256i mask_a = _mm256_cmpgt_epi16(v_th, sum_a);
        __m256i mask_a_high = _mm256_cmpgt_epi16(v_th, sum_a_high);

        __m256i mask_b = _mm256_cmpgt_epi16(v_th, sum_b);
        __m256i mask_b_high = _mm256_cmpgt_epi16(v_th, sum_b_high);

        __m256i mask_c = _mm256_cmpgt_epi16(v_th, sum_c);
        __m256i mask_c_high = _mm256_cmpgt_epi16(v_th, sum_c_high);

        // Combine masks for each register
        __m256i mask8_a = _mm256_packus_epi16(mask_a, mask_a_high);
        __m256i mask8_b = _mm256_packus_epi16(mask_b, mask_b_high);
        __m256i mask8_c = _mm256_packus_epi16(mask_c, mask_c_high);

        // Apply the mask to whiten pixels
        a = _mm256_blendv_epi8(a, v_cn, mask8_a);
        b = _mm256_blendv_epi8(b, v_cn, mask8_b);
        c = _mm256_blendv_epi8(c, v_cn, mask8_c);

        // Store the results back to memory
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a);
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 32), b);
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 64), c);

    }

    // Process remaining pixels with a scalar loop
    for (; p < data + size; p += 3)
    {
        if (p[0] + p[1] + p[2] < threshold)
        {
            p[0] = p[1] = p[2] = 255 - cn;
        }
    }

    return *this;
}

Image& Image::darkenAboveThreshold_ColorNuance_AVX2(const int threshold, const std::uint8_t cn)
{
    // Set the constant color value for darkening
    const __m256i v_cn = _mm256_set1_epi8(static_cast<char>(cn));
    // Set the threshold value for comparison
	const __m256i v_th = _mm256_set1_epi8(static_cast<char>(threshold));

    std::uint8_t* p = data;
    std::uint8_t* end = data + size - 96; // 32 RGB pixels (96 bytes)

    // Process 32 pixels at a time using AVX2
    for (; p <= end; p += 96)
    {
        // Load 96 bytes (32 RGB pixels) into three 256-bit registers
    	__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    	__m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    	__m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64));

        // Extract R, G, B components for the first 32 bytes (a)
        __m256i r_a = _mm256_and_si256(a, _mm256_set1_epi32(0xFF));
        __m256i g_a = _mm256_and_si256(_mm256_srli_epi32(a, 8), _mm256_set1_epi32(0xFF));
        __m256i b_a = _mm256_and_si256(_mm256_srli_epi32(a, 16), _mm256_set1_epi32(0xFF));

        // Extract R, G, B components for the next 32 bytes (b)
        __m256i r_b = _mm256_and_si256(b, _mm256_set1_epi32(0xFF));
        __m256i g_b = _mm256_and_si256(_mm256_srli_epi32(b, 8), _mm256_set1_epi32(0xFF));
        __m256i b_b = _mm256_and_si256(_mm256_srli_epi32(b, 16), _mm256_set1_epi32(0xFF));

        // Extract R, G, B components for the last 32 bytes (c)
        __m256i r_c = _mm256_and_si256(c, _mm256_set1_epi32(0xFF));
        __m256i g_c = _mm256_and_si256(_mm256_srli_epi32(c, 8), _mm256_set1_epi32(0xFF));
        __m256i b_c = _mm256_and_si256(_mm256_srli_epi32(c, 16), _mm256_set1_epi32(0xFF));

        // Convert R, G, B components to 16-bit for the first 16 pixels (lower 128 bits)
        __m256i r16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_a));
        __m256i g16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_a));
        __m256i b16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_a));

        __m256i r16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_b));
        __m256i g16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_b));
        __m256i b16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_b));

        __m256i r16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_c));
        __m256i g16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_c));
        __m256i b16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_c));

        // Convert R, G, B components to 16-bit for the next 16 pixels (upper 128 bits)
        __m256i r16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_a, 1));
        __m256i g16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_a, 1));
        __m256i b16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_a, 1));

        __m256i r16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_b, 1));
        __m256i g16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_b, 1));
        __m256i b16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_b, 1));

        __m256i r16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_c, 1));
        __m256i g16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_c, 1));
        __m256i b16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_c, 1));

    	// Calculate the sum of R + G + B for each group of 16 pixels (unsigned saturated)
    	__m256i sum_a = _mm256_adds_epu16(_mm256_adds_epu16(r16_a, g16_a), b16_a);
    	__m256i sum_a_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_a_high, g16_a_high), b16_a_high);

    	__m256i sum_b = _mm256_adds_epu16(_mm256_adds_epu16(r16_b, g16_b), b16_b);
    	__m256i sum_b_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_b_high, g16_b_high), b16_b_high);

    	__m256i sum_c = _mm256_adds_epu16(_mm256_adds_epu16(r16_c, g16_c), b16_c);
    	__m256i sum_c_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_c_high, g16_c_high), b16_c_high);


        // Compare the sum to the threshold
    	__m256i mask_a = _mm256_cmpgt_epi16(sum_a, v_th);  // sum > threshold
    	__m256i mask_a_high = _mm256_cmpgt_epi16(sum_a_high, v_th);

        __m256i mask_b = _mm256_cmpgt_epi16(sum_b, v_th);
        __m256i mask_b_high = _mm256_cmpgt_epi16(sum_b_high, v_th);

        __m256i mask_c = _mm256_cmpgt_epi16(sum_c, v_th);
        __m256i mask_c_high = _mm256_cmpgt_epi16(sum_c_high, v_th);

        // Combine masks for each register
        __m256i mask8_a = _mm256_packus_epi16(mask_a, mask_a_high);
        __m256i mask8_b = _mm256_packus_epi16(mask_b, mask_b_high);
        __m256i mask8_c = _mm256_packus_epi16(mask_c, mask_c_high);

        // Apply the mask to darken pixels
        a = _mm256_blendv_epi8(a, v_cn, mask8_a);
        b = _mm256_blendv_epi8(b, v_cn, mask8_b);
        c = _mm256_blendv_epi8(c, v_cn, mask8_c);

        // Store the results back to memory
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a);
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 32), b);
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 64), c);

    }

    // Process remaining pixels with a scalar loop
    for (; p < data + size; p += 3)
    {
        if (p[0] + p[1] + p[2] > threshold)
        {
            p[0] = p[1] = p[2] = cn;
        }
    }

    return *this;
}

Image& Image::whitenAboveThreshold_ColorNuance_AVX2(const int threshold, const std::uint8_t cn)
{
    // Set the constant color value for darkening
    const __m256i v_cn = _mm256_set1_epi8(static_cast<char>(255 - cn));
    // Set the threshold value for comparison
	const __m256i v_th = _mm256_set1_epi8(static_cast<char>(threshold));

    std::uint8_t* p = data;
    std::uint8_t* end = data + size - 96; // 32 RGB pixels (96 bytes)

    // Process 32 pixels at a time using AVX2
    for (; p <= end; p += 96)
    {
        // Load 96 bytes (32 RGB pixels) into three 256-bit registers
    	__m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    	__m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
    	__m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 64));

        // Extract R, G, B components for the first 32 bytes (a)
        __m256i r_a = _mm256_and_si256(a, _mm256_set1_epi32(0xFF));
        __m256i g_a = _mm256_and_si256(_mm256_srli_epi32(a, 8), _mm256_set1_epi32(0xFF));
        __m256i b_a = _mm256_and_si256(_mm256_srli_epi32(a, 16), _mm256_set1_epi32(0xFF));

        // Extract R, G, B components for the next 32 bytes (b)
        __m256i r_b = _mm256_and_si256(b, _mm256_set1_epi32(0xFF));
        __m256i g_b = _mm256_and_si256(_mm256_srli_epi32(b, 8), _mm256_set1_epi32(0xFF));
        __m256i b_b = _mm256_and_si256(_mm256_srli_epi32(b, 16), _mm256_set1_epi32(0xFF));

        // Extract R, G, B components for the last 32 bytes (c)
        __m256i r_c = _mm256_and_si256(c, _mm256_set1_epi32(0xFF));
        __m256i g_c = _mm256_and_si256(_mm256_srli_epi32(c, 8), _mm256_set1_epi32(0xFF));
        __m256i b_c = _mm256_and_si256(_mm256_srli_epi32(c, 16), _mm256_set1_epi32(0xFF));

        // Convert R, G, B components to 16-bit for the first 16 pixels (lower 128 bits)
        __m256i r16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_a));
        __m256i g16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_a));
        __m256i b16_a = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_a));

        __m256i r16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_b));
        __m256i g16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_b));
        __m256i b16_b = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_b));

        __m256i r16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_c));
        __m256i g16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_c));
        __m256i b16_c = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_c));

        // Convert R, G, B components to 16-bit for the next 16 pixels (upper 128 bits)
        __m256i r16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_a, 1));
        __m256i g16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_a, 1));
        __m256i b16_a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_a, 1));

        __m256i r16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_b, 1));
        __m256i g16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_b, 1));
        __m256i b16_b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_b, 1));

        __m256i r16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(r_c, 1));
        __m256i g16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(g_c, 1));
        __m256i b16_c_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_c, 1));

    	// Calculate the sum of R + G + B for each group of 16 pixels (unsigned saturated)
    	__m256i sum_a = _mm256_adds_epu16(_mm256_adds_epu16(r16_a, g16_a), b16_a);
    	__m256i sum_a_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_a_high, g16_a_high), b16_a_high);

    	__m256i sum_b = _mm256_adds_epu16(_mm256_adds_epu16(r16_b, g16_b), b16_b);
    	__m256i sum_b_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_b_high, g16_b_high), b16_b_high);

    	__m256i sum_c = _mm256_adds_epu16(_mm256_adds_epu16(r16_c, g16_c), b16_c);
    	__m256i sum_c_high = _mm256_adds_epu16(_mm256_adds_epu16(r16_c_high, g16_c_high), b16_c_high);


        // Compare the sum to the threshold
        __m256i mask_a = _mm256_cmpgt_epi16(sum_a, v_th);
        __m256i mask_a_high = _mm256_cmpgt_epi16(sum_a_high, v_th);

        __m256i mask_b = _mm256_cmpgt_epi16(sum_b, v_th);
        __m256i mask_b_high = _mm256_cmpgt_epi16(sum_b_high, v_th);

        __m256i mask_c = _mm256_cmpgt_epi16(sum_c, v_th);
        __m256i mask_c_high = _mm256_cmpgt_epi16(sum_c_high, v_th);

        // Combine masks for each register
        __m256i mask8_a = _mm256_packus_epi16(mask_a, mask_a_high);
        __m256i mask8_b = _mm256_packus_epi16(mask_b, mask_b_high);
        __m256i mask8_c = _mm256_packus_epi16(mask_c, mask_c_high);

        // Apply the mask to whiten pixels
        a = _mm256_blendv_epi8(a, v_cn, mask8_a);
        b = _mm256_blendv_epi8(b, v_cn, mask8_b);
        c = _mm256_blendv_epi8(c, v_cn, mask8_c);

        // Store the results back to memory
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p), a);
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 32), b);
    	_mm256_storeu_si256(reinterpret_cast<__m256i*>(p + 64), c);

    }

    // Process remaining pixels with a scalar loop
    for (; p < data + size; p += 3)
    {
        if (p[0] + p[1] + p[2] > threshold)
        {
            p[0] = p[1] = p[2] = 255 - cn;
        }
    }

    return *this;
}

Image& Image::operator=(const Image& img) {
	if (this == &img) {
		return *this;  // Protection auto-affectation
	}

	// Libération de l'ancien buffer
	delete[] data;

	// Copie des nouvelles dimensions
	w = img.w;
	h = img.h;
	channels = img.channels;
	size = img.size;

	// Allocation et copie du nouveau buffer
	data = new uint8_t[size];
	memcpy(data, img.data, size);

	return *this;
}


Image& Image::std_convolve_clamp_to_0(const int channel, const int ker_w, const int ker_h, const double *ker, int cr, int c) {

    // Vérification des dimensions de l'image
    if (w <= 0 || h <= 0 || data == nullptr) {
        fprintf(stderr, "Erreur : Image invalide (dimensions ou données incorrectes).\n");
        return *this;
    }

    // Allocation dynamique du buffer temporaire
    auto *new_data = new uint8_t[w * h];
    if (new_data == nullptr) {
        fprintf(stderr, "Erreur : Échec de l'allocation mémoire pour new_data.\n");
        return *this;
    }

    // Initialisation du buffer temporaire
    memset(new_data, 0, w * h);

    // Calcul de la convolution
    const int half_ker_w = ker_w / 2;
    const int half_ker_h = ker_h / 2;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double sum = 0.0;

            // Application du noyau de convolution
            for (int ky = -half_ker_h; ky <= half_ker_h; ++ky) {
                for (int kx = -half_ker_w; kx <= half_ker_w; ++kx) {
                    const int pixel_x = x + kx;
                    const int pixel_y = y + ky;

                    // Gestion des bords : clamp to 0
                    if (pixel_x < 0 || pixel_x >= w || pixel_y < 0 || pixel_y >= h) {
                        continue;
                    }

                    // Calcul de l'indice du pixel
                    const int idx = pixel_y * w + pixel_x;

                    // Calcul de la contribution du pixel au résultat
                    const int ker_idx = (ky + half_ker_h) * ker_w + (kx + half_ker_w);
                    sum += data[channels * idx + channel] * ker[ker_idx];
                }
            }

            // Stockage du résultat dans new_data
            const int idx = y * w + x;
            new_data[idx] = static_cast<uint8_t>(sum);
        }
    }

    // Copie du résultat dans this > data
    for (uint64_t i = 0; i < static_cast<uint64_t>(w) * h; ++i) {
        data[channels * i + channel] = new_data[i];
    }

    // Libération de la mémoire
    delete[] new_data;

    return *this;
}



Image& Image::std_convolve_clamp_to_border(const uint8_t channel, const uint32_t ker_w,
	const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	uint8_t new_data[w*h];
	const uint64_t center = cr*ker_w + cc;
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for (long i = -static_cast<long>(cr); i < static_cast<long>(ker_h - cr); ++i) {
			long row = (static_cast<long>(k / channels) / w) - i;
			if(row < 0) {
				row = 0;
			} else if(row > h-1) {
				row = h-1;
			}
			for (long j = -static_cast<long>(cc); j < static_cast<long>(ker_w - cc); ++j) {
				long col = (static_cast<long>(k / channels) % w) - j;
				if(col < 0) {
					col = 0;
				} else if(col > w-1) {
					col = w-1;
				}
				uint64_t ker_idx = center + i * static_cast<long>(ker_w) + j;
				c += ker[ker_idx] * data[(row * w + col) * channels + channel];
			}
		}
		new_data[k / channels] = static_cast<uint8_t>(BYTE_BOUND(round(c)));
	}
	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}


Image& Image::std_convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	uint8_t new_data[w*h];
	uint64_t center = cr*ker_w + cc;
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for(long i = -(static_cast<long>(cr)); i<static_cast<long>(ker_h)-cr; ++i) {
			long row = (static_cast<long>(k)/channels)/w-i;
			if(row < 0) {
				row = row%h + h;
			} else if(row > h-1) {
				row %= h;
			}
			for(long j = -(static_cast<long>(cc)); j<static_cast<long>(ker_w)-cc; ++j) {
				long col = (static_cast<long>(k)/channels)%w-j;
				if(col < 0) {
					col = col%w + w;
				} else if(col > w-1) {
					col %= w;
				}
				c += ker[center+i*static_cast<long>(ker_w)+j]*data[(row*w+col)*channels+channel];
			}
		}
		new_data[k/channels] = static_cast<uint8_t>(BYTE_BOUND(round(c)));

	}
	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}






uint32_t Image::rev(uint32_t n, uint32_t a) {
	auto max_bits = static_cast<uint8_t>(ceil(log2(n)));
	uint32_t reversed_a = 0;
	for(uint8_t i=0; i<max_bits; ++i) {
		if(a & (1<<i)) {
			reversed_a |= (1<<(max_bits-1-i));
		}
	}
	return reversed_a;
}

void Image::bit_rev(uint32_t n, std::complex<double> a[], std::complex<double>* A) {
	for(uint32_t i=0; i<n; ++i) {
		A[rev(n,i)] = a[i];
	}
}

void Image::fft(uint32_t n, std::complex<double> x[], std::complex<double>* X) {
	// x in standard order
	if (x != X) {
		memcpy(X, x, n * sizeof(std::complex<double>));
	}

	// Gentleman-Sande butterfly
	uint32_t sub_probs = 1;
	uint32_t sub_prob_size = n;

	while (sub_prob_size > 1) {
		uint32_t half = sub_prob_size >> 1;
		std::complex<double> w_step(cos(-2 * M_PI / sub_prob_size), sin(-2 * M_PI / sub_prob_size));

		for (uint32_t i = 0; i < sub_probs; ++i) {
			uint32_t j_begin = i * sub_prob_size;
			uint32_t j_end = j_begin + half;
			std::complex<double> w(1, 0);

			for (uint32_t j = j_begin; j < j_end; ++j) {
				std::complex<double> tmp1 = X[j];
				std::complex<double> tmp2 = X[j + half];
				X[j] = tmp1 + tmp2;
				X[j + half] = (tmp1 - tmp2) * w;
				w *= w_step;
			}
		}

		sub_probs <<= 1;
		sub_prob_size = half;
	}
	// X in bit reversed order
}


void Image::ifft(uint32_t n, std::complex<double> X[], std::complex<double>* x) {
	// X in bit reversed order
	if (X != x) {
		memcpy(x, X, n * sizeof(std::complex<double>));
	}

	// Cooley-Tukey butterfly
	uint32_t sub_probs = n >> 1;
	uint32_t half = 1;

	while (half < n) {
		uint32_t sub_prob_size = half << 1;
		std::complex<double> w_step(cos(2 * M_PI / sub_prob_size), sin(2 * M_PI / sub_prob_size));

		for (uint32_t i = 0; i < sub_probs; ++i) {
			uint32_t j_begin = i * sub_prob_size;
			uint32_t j_end = j_begin + half;
			std::complex<double> w(1, 0);

			for (uint32_t j = j_begin; j < j_end; ++j) {
				std::complex<double> tmp1 = x[j];
				std::complex<double> tmp2 = w * x[j + half];
				x[j] = tmp1 + tmp2;
				x[j + half] = tmp1 - tmp2;
				w *= w_step;
			}
		}

		sub_probs >>= 1;
		half = sub_prob_size;
	}

	for (uint32_t i = 0; i < n; ++i) {
		x[i] /= n;
	}
	// x in standard order
}


void Image::dft_2D(uint32_t m, uint32_t n, std::complex<double> x[], std::complex<double>* X) {
	//x in row-major & standard order
	auto* intermediate = new std::complex<double>[m*n];
	//rows
	for(uint32_t i=0; i<m; ++i) {
		fft(n, x+i*n, intermediate+i*n);
	}
	//cols
	for(uint32_t j=0; j<n; ++j) {
		for(uint32_t i=0; i<m; ++i) {
			X[j*m+i] = intermediate[i*n+j]; //row-major --> col-major
		}
		fft(m, X+j*m, X+j*m);
	}
	delete[] intermediate;
	//X in column-major & bit-reversed (in rows then columns)
}

void Image::idft_2D(uint32_t m, uint32_t n, std::complex<double> X[], std::complex<double>* x) {
	//X in column-major & bit-reversed (in rows then columns)
	auto* intermediate = new std::complex<double>[m*n];
	//cols
	for(uint32_t j=0; j<n; ++j) {
		ifft(m, X+j*m, intermediate+j*m);
	}
	//rows
	for(uint32_t i=0; i<m; ++i) {
		for(uint32_t j=0; j<n; ++j) {
			x[i*n+j] = intermediate[j*m+i]; //row-major <-- col-major
		}
		ifft(n, x+i*n, x+i*n);
	}
	delete[] intermediate;
	//x in row-major & standard order
}

void Image::pad_kernel(const uint32_t ker_w, const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc, const uint32_t pw, const uint32_t ph, std::complex<double>* pad_ker) {
	//padded so center of kernel is at top left
	for(long i=-(static_cast<long>(cr)); i<static_cast<long>(ker_h)-cr; ++i) {
		const uint32_t r = (i<0) ? i+ph : i;
		for(long j=-(static_cast<long>(cc)); j<static_cast<long>(ker_w)-cc; ++j) {
			uint32_t c = (j<0) ? j+pw : j;
			pad_ker[r*pw+c] = std::complex<double>(ker[(i+cr)*ker_w+(j+cc)], 0);
		}
	}
}
void Image::pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double>* p) {
	for(uint64_t k=0; k<l; ++k) {
		p[k] = a[k]*b[k];
	}
}

Image& Image::fd_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	//calculate padding
	/* 1.0
	uint32_t pw = 1<<((uint8_t)ceil(log2(w+ker_w-1)));
	uint32_t ph = 1<<((uint8_t)ceil(log2(h+ker_h-1)));
	uint64_t psize = pw*ph;
	*/
	uint32_t pw = 1 << static_cast<uint8_t>(ceil(log2(static_cast<double>(w + ker_w - 1))));
	uint32_t ph = 1 << static_cast<uint8_t>(ceil(log2(static_cast<double>(h + ker_h - 1))));
	uint64_t psize = static_cast<uint64_t>(pw) * static_cast<uint64_t>(ph);

	//pad image
	auto* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			pad_img[i*pw+j] = std::complex<double>(data[(i*w+j)*channels+channel],0);
		}
	}

	//pad kernel
	auto* pad_ker = new std::complex<double>[psize];
	pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

	//convolution
	dft_2D(ph, pw, pad_img, pad_img);
	dft_2D(ph, pw, pad_ker, pad_ker);
	pointwise_product(psize, pad_img, pad_ker, pad_img);
	idft_2D(ph, pw, pad_img, pad_img);

	//update pixel data
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			data[(i*w+j)*channels+channel] = BYTE_BOUND((uint8_t)round(pad_img[i*pw+j].real()));
		}
	}

	return *this;
}
Image& Image::fd_convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	//calculate padding
	const uint32_t pw = 1 << (static_cast<uint8_t>(ceil(log2(w + ker_w - 1))));
	const uint32_t ph = 1 << (static_cast<uint8_t>(ceil(log2(h+ker_h-1))));
	uint64_t psize = pw*ph;

	//pad image
	auto* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<ph; ++i) {
		uint32_t r = (i<h) ? i : ((i<h+cr ? h-1 : 0));
		for(uint32_t j=0; j<pw; ++j) {
			uint32_t c = (j<w) ? j : ((j<w+cc ? w-1 : 0));
			pad_img[i*pw+j] = std::complex<double>(data[(r*w+c)*channels+channel],0);
		}
	}

	//pad kernel
	auto* pad_ker = new std::complex<double>[psize];
	pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

	//convolution
	dft_2D(ph, pw, pad_img, pad_img);
	dft_2D(ph, pw, pad_ker, pad_ker);
	pointwise_product(psize, pad_img, pad_ker, pad_img);
	idft_2D(ph, pw, pad_img, pad_img);

	//update pixel data
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			data[(i*w+j)*channels+channel] = BYTE_BOUND((uint8_t)round(pad_img[i*pw+j].real()));
		}
	}

	return *this;
}

Image& Image::fd_convolve_cyclic(const uint8_t channel, const uint32_t ker_w,
	const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	//calculate padding
	const uint32_t pw = 1 << (static_cast<uint8_t>(ceil(log2(w + ker_w - 1))));
	const uint32_t ph = 1 << (static_cast<uint8_t>(ceil(log2(h + ker_h - 1))));

	uint64_t psize = pw*ph;

	//pad image
	auto* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<ph; ++i) {
		uint32_t r = (i<h) ? i : ((i<h+cr ? i%h : h-ph+i));
		for(uint32_t j=0; j<pw; ++j) {
			uint32_t c = (j<w) ? j : ((j<w+cc ? j%w : w-pw+j));
			pad_img[i*pw+j] = std::complex<double>(data[(r*w+c)*channels+channel],0);
		}
	}

	//pad kernel
	auto* pad_ker = new std::complex<double>[psize];
	pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

	//convolution
	dft_2D(ph, pw, pad_img, pad_img);
	dft_2D(ph, pw, pad_ker, pad_ker);
	pointwise_product(psize, pad_img, pad_ker, pad_img);
	idft_2D(ph, pw, pad_img, pad_img);

	//update pixel data
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			data[(i*w+j)*channels+channel] = BYTE_BOUND((uint8_t)round(pad_img[i*pw+j].real()));
		}
	}

	return *this;
}


Image& Image::convolve_linear(const uint8_t channel, const uint32_t ker_w, const uint32_t ker_h,
	double ker[], const uint32_t cr, const uint32_t cc) {
	if (static_cast<uint64_t>(ker_w) * static_cast<uint64_t>(ker_h) > 224) {
		return fd_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	} else {
		return std_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	}
}

Image& Image::convolve_clamp_to_border(const uint8_t channel, const uint32_t ker_w,
	const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_clamp_to_border(channel, ker_w, ker_h, ker, cr, cc);
	} else {
		return std_convolve_clamp_to_border(channel, ker_w, ker_h, ker, cr, cc);
	}
}
Image& Image::convolve_cyclic(const uint8_t channel, const  uint32_t ker_w, const uint32_t ker_h, double ker[], const uint32_t cr, const uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_cyclic(channel, ker_w, ker_h, ker, cr, cc);
	} else {
		return std_convolve_cyclic(channel, ker_w, ker_h, ker, cr, cc);
	}
}


Image& Image::diffmap(const Image& img) {
	const int compare_width = std::min(w, img.w);
	const int compare_height = std::min(h, img.h);
	const int compare_channels = std::min(channels, img.channels);

	for (uint32_t i = 0; i < compare_height; ++i) {
		for (uint32_t j = 0; j < compare_width; ++j) {
			for (int k = 0; k < compare_channels; ++k) { // Changed uint8_t to int
				data[(i * w + j) * channels + k] = static_cast<uint8_t>(BYTE_BOUND(abs(data[(i * w + j) * channels + k] - img.data[(i * img.w + j) * img.channels + k])));
			}
		}
	}
	return *this;
}



Image& Image::diffmap_scale(Image& img, uint8_t scl) {
	const int compare_width = std::min(w, img.w);
	const int compare_height = std::min(h, img.h);
	const int compare_channels = std::min(channels, img.channels);

	uint8_t largest = 0;
	for(uint32_t i=0; i<compare_height; ++i) {
		for(uint32_t j=0; j<compare_width; ++j) {
			for(uint8_t k=0; k<compare_channels; ++k) {
				data[(i*w+j)*channels+k] = BYTE_BOUND(abs(data[(i*w+j)*channels+k] - img.data[(i*img.w+j)*img.channels+k]));
				largest = fmax(largest, data[(i*w+j)*channels+k]);
			}
		}
	}
	scl = static_cast<uint8_t>(255 / fmax(1, fmax(scl, largest)));
	for(int i=0; i<size; ++i) {
		data[i] *= scl;
	}
	return *this;
}


Image& Image::grayscale_avg() {
	if(channels < 3) {
		printf("Image %p has less than 3 channels, it is assumed to already be grayscale.", this);
	} else {
		for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
			//(r+g+b)/3
			int gray = (data[i] + data[i+1] + data[i+2])/3;
			memset(data+i, gray, 3);
		}
	}
	return *this;
}


Image& Image::grayscale_lum() {
	if(channels < 3) {
		printf("Image %p has less than 3 channels, it is assumed to already be grayscale.", this);
	} else {
		for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
			const double gray_d = 0.2126*data[i] + 0.7152*data[i+1] + 0.0722*data[i+2];
			const int gray = static_cast<int>(std::round(gray_d));
			memset(data+i, gray, 3);
		}
	}
	return *this;
}

Image& Image::color_mask(const float r, const float g, const float b) {
	if (channels < 3) {
		printf("\e[31m[ERROR] Color mask requires at least 3 channels, but this image has %d channels\e[0m\n", channels);
	} else {
		for (size_t i = 0; i < size; i += static_cast<size_t>(channels)) {
			data[i]   = static_cast<uint8_t>(std::clamp(std::round(static_cast<float>(data[i])   * r), 0.0f, 255.0f));
			data[i+1] = static_cast<uint8_t>(std::clamp(std::round(static_cast<float>(data[i+1]) * g), 0.0f, 255.0f));
			data[i+2] = static_cast<uint8_t>(std::clamp(std::round(static_cast<float>(data[i+2]) * b), 0.0f, 255.0f));
		}
	}
	return *this;
}


Image& Image::encodeMessage(const char* message) {
	const uint32_t len = strlen(message) * 8;
	if(len + STEG_HEADER_SIZE > size) {
		printf("\e[31m[ERROR] This message is too large (%lu bits / %zu bits)\e[0m\n", len+STEG_HEADER_SIZE, size);
		return *this;
	}

	for(size_t i = 0; i < STEG_HEADER_SIZE; ++i) {
		data[i] &= 0xFE;
		data[i] |= (len >> (STEG_HEADER_SIZE - 1 - i)) & 1UL;
	}

	for(uint32_t i = 0; i < len; ++i) {
		data[i+STEG_HEADER_SIZE] &= 0xFE;
		data[i+STEG_HEADER_SIZE] |= (message[i/8] >> ((len-1-i)%8)) & 1;
	}

	return *this;
}

Image& Image::decodeMessage(char* buffer, size_t* messageLength) {
	constexpr uint32_t len = 0;
	for(size_t i = 0; i < STEG_HEADER_SIZE; ++i) {
		data[i] &= 0xFE;
		data[i] |= (len >> (STEG_HEADER_SIZE - 1 - i)) & 1UL;
	}

	*messageLength = len / 8;

	for(uint32_t i = 0; i < len; ++i) {
		buffer[i/8] = static_cast<char>((buffer[i/8] << 1) | (data[i+STEG_HEADER_SIZE] & 1));
	}


	return *this;
}




Image& Image::flipX() {
	uint8_t tmp[4];
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w / 2; ++x) {
			uint8_t* px1 = &data[(x + y * w) * channels];
			uint8_t* px2 = &data[((w - 1 - x) + y * w) * channels];

			// Utilisation de std::copy_n pour copier `channels` éléments
			std::copy_n(px1, channels, tmp);
			std::copy_n(px2, channels, px1);
			std::copy_n(tmp, channels, px2);
		}
	}
	return *this;
}


Image& Image::flipY() {
	uint8_t tmp[4];
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w / 2; ++x) {
			uint8_t* px1 = &data[(x + y * w) * channels];
			uint8_t* px2 = &data[((w - 1 - x) + y * w) * channels];

			// Utilisation de std::copy_n pour copier `channels` éléments
			std::copy_n(tmp, channels, px1);
			std::copy_n(px1, channels, px2);
			std::copy_n(px2, channels, tmp);
		}
	}
	return *this;
}


Image& Image::overlay(const Image& source, const int x, const int y) {
	for (int sy = 0; sy < source.h; ++sy) {
		if (sy + y < 0) continue;
		if (sy + y >= h) break;

		for (int sx = 0; sx < source.w; ++sx) {
			if (sx + x < 0) continue;
			if (sx + x >= w) break;

			uint8_t* srcPx = &source.data[(sx + sy * source.w) * source.channels];
			uint8_t* dstPx = &data[(sx + x + (sy + y) * w) * channels];

			const float srcAlpha = (source.channels < 4) ? 1.0f : (static_cast<float>(srcPx[3]) / 255.0f);
			const float dstAlpha = (channels < 4) ? 1.0f : (static_cast<float>(dstPx[3]) / 255.0f);


			if (srcAlpha > 0.99f && dstAlpha > 0.99f) {
				if (source.channels >= channels) {
					std::copy(srcPx, srcPx + channels, dstPx);
				} else {
					std::fill(dstPx, dstPx + channels, srcPx[0]);
				}
			} else {
				float outAlpha = srcAlpha + dstAlpha * (1.0f - srcAlpha);
				if (outAlpha < 0.01f) {
					std::fill(dstPx, dstPx + channels, 0);
				} else {
					for (int chnl = 0; chnl < channels; ++chnl) {
						dstPx[chnl] = static_cast<uint8_t>(
							BYTE_BOUND((srcPx[chnl] / 255.0f * srcAlpha +
									   dstPx[chnl] / 255.0f * dstAlpha * (1.0f - srcAlpha)) /
									  outAlpha * 255.0f)
						);
					}
					if (channels > 3) {
						dstPx[3] = static_cast<uint8_t>(BYTE_BOUND(outAlpha * 255.0f));
					}
				}
			}
		}
	}
	return *this;
}


Image& Image::crop(const uint16_t cx, const uint16_t cy, const uint16_t cw, const uint16_t ch) {
	size = cw * ch * channels;
	auto* croppedImage = new uint8_t[size];
	memset(croppedImage, 0, size);

	for(uint16_t y = 0; y < ch; ++y) {
		if(y + cy >= h) {
			break;
		}
		for(uint16_t x = 0; x < cw; ++x) {
			if(x + cx >= w) {
				break;
			}
			memcpy(&croppedImage[(x + y * cw) * channels], &data[(x + cx + (y + cy) * w) * channels], channels);
		}
	}

	w = cw;
	h = ch;


	delete[] data;
	data = croppedImage;
	croppedImage = nullptr;

	return *this;
}




Image& Image::resizeNN(const uint16_t nw, const uint16_t nh) {
	size = nw * nh * channels;
	auto* newImage = new uint8_t[size];

	const float scaleX = static_cast<float>(nw) / static_cast<float>(w);
	const float scaleY = static_cast<float>(nh) / static_cast<float>(h);

	for(uint16_t y = 0; y < nh; ++y) {
		const auto sy = static_cast<uint16_t>(std::round(y / scaleY));
		for(uint16_t x = 0; x < nw; ++x) {
			const auto sx = static_cast<uint16_t>(std::round(x / scaleX));


			memcpy(&newImage[(x + y * nw) * channels], &data[(sx + sy * w) * channels], channels);

		}
	}

	w = nw;
	h = nh;
	delete[] data;
	data = newImage;
	newImage = nullptr;

	return *this;
}
