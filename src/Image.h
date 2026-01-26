#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>
#include <cstdio>
#include <complex>
#include <vector>


//legacy feature of C
#undef __STRICT_ANSI__
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#ifndef M_PI
	#define M_PI (3.14159265358979323846)
#endif

#include "schrift.h"

#define STEG_HEADER_SIZE sizeof(uint32_t) * 8


enum ImageType {
	PNG, JPG, BMP, TGA
};

struct Font;

namespace SimpleColors {
	constexpr uint8_t ONE_THIRD = 85;
	constexpr uint8_t HALF = 127;
	constexpr uint8_t FULL = 255;
}


struct Image {
	uint8_t* data = nullptr;
	size_t size = 0;
	int w;
	int h;
	int channels;

	// Structure pour stocker les indices des canaux
	struct ChannelIndices {
		int max;
		int mid;
		int min;
	};

	// Fonction helper pour déterminer l'ordre des canaux de couleur
	static ChannelIndices get_channel_indices(const uint8_t r, const  uint8_t g, const uint8_t b) {
		if (r > g && g > b) return {0, 1, 2};
		if (r > b && b > g) return {0, 2, 1};
		if (g > r && r > b) return {1, 0, 2};
		if (g > b && b > r) return {1, 2, 0};
		if (b > r && r > g) return {2, 0, 1};
		return {2, 1, 0}; // Cas par défaut
	}

	Image(const char* filename, int channel_force = 0);
	Image(int w, int h, int channels = 3);
	Image(const Image& img);
	Image& operator=(const Image& img);
	~Image();

	bool read(const char* filename, int channel_force = 0);
	bool write(const char* filename);

	static ImageType get_file_type(const char* filename);

	Image& std_convolve_clamp_to_0(int channel, int ker_w, int ker_h, const double *ker, int cr, int c);
	Image& std_convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& std_convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);


	static bool approx_equal(const uint8_t a, const  uint8_t b, const uint8_t tol) {
		const int diff = static_cast<int>(a) - static_cast<int>(b);
		return diff >= 0 ? diff <= tol : -diff <= tol;
	}

	static uint8_t avg_u8_round(const uint8_t a, const uint8_t b) {
		return static_cast<uint8_t>(
			(static_cast<unsigned int>(a) +
			 static_cast<unsigned int>(b) + 1u) / 2u
		);
	}

	static uint32_t rev(uint32_t n, uint32_t a);
	static void bit_rev(uint32_t n, std::complex<double> a[], std::complex<double>* A);

	static void fft(uint32_t n, std::complex<double> x[], std::complex<double>* X);
	static void ifft(uint32_t n, std::complex<double> X[], std::complex<double>* x);
	static void dft_2D(uint32_t m, uint32_t n, std::complex<double> x[], std::complex<double>* X);
	static void idft_2D(uint32_t m, uint32_t n, std::complex<double> X[], std::complex<double>* x);

	static void pad_kernel(uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double>* pad_ker);
	static inline void pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double>* p);

	Image& fd_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& fd_convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& fd_convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);


	Image& convolve_linear(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);


	Image& diffmap(const Image& img);
	Image& diffmap_scale(Image& img, uint8_t scl = 0);


	Image& grayscale_avg();

	Image& below_threshold(int threshold, int cn, bool useDarkNuance);
	Image& aboveThreshold(int threshold, int cn, bool useDarkNuance);
	Image& belowProportion(float proportion, int cn, bool useDarkNuance);
	Image& aboveProportion(float proportion, int cn, bool useDarkNuance);

	Image& darkenBelowThreshold_ColorNuance_AVX2(int threshold, uint8_t cn);
	Image& whitenBelowThreshold_ColorNuance_AVX2(int threshold, uint8_t cn);
	Image& darkenAboveThreshold_ColorNuance_AVX2(int threshold, uint8_t cn);
	Image& whitenAboveThreshold_ColorNuance_AVX2(int threshold, uint8_t cn);


	Image& reverseBelowThreshold(int threshold);
	Image& reverseAboveThreshold(int threshold);

	Image& alternatelyDarkenAndWhitenBelowTheThreshold(int s, int first_threshold,	int last_threshold);
	Image& alternatelyDarkenAndWhitenAboveTheThreshold(int s, int first_threshold,	int last_threshold);

	Image& original_black_and_white(int threshold);
	Image& reversed_black_and_white(int threshold);
	
	static void simplify_pixel(
	uint8_t& r, uint8_t& g, uint8_t& b,
	uint8_t r_val_third, uint8_t g_val_third, uint8_t b_val_third,
	uint8_t r_val_half, uint8_t g_val_half, uint8_t b_val_half,
	uint8_t r_val_full, uint8_t g_val_full, uint8_t b_val_full,
	int tolerance);

	Image& simplify_to_dominant_color_combinations_with_average(int tolerance);
	Image& simplify_to_dominant_color_combinations_without_average(int tolerance);

	template<typename ConditionFunc, typename TransformFunc>
	Image& applyThresholdTransformationRegionFraction(
		int threshold,
		int fraction,
		const std::vector<int>& rectanglesToModify,
		ConditionFunc condition,
		TransformFunc transformation
	);

	Image& darkenBelowThresholdRegionFraction(int threshold, int cn, int fraction,
		const std::vector<int>& rectanglesToModify);
	Image& whitenBelowThresholdRegionFraction(int threshold, int cn, int fraction,
		const std::vector<int>& rectanglesToModify);
	Image& darkenAboveThresholdRegionFraction(int threshold, int cn, int fraction,
		const std::vector<int>& rectanglesToModify);
	Image& whitenAboveThresholdRegionFraction(int threshold, int cn, int fraction,
		const std::vector<int>& rectanglesToModify);


	Image& grayscale_lum();

	Image& color_mask(float r, float g, float b);


	Image& encodeMessage(const char* message);
	Image& decodeMessage(char* buffer, size_t* messageLength);

	Image& flipX();
	Image& flipY();

	Image& overlay(const Image& source, int x, int y);

	Image& overlayText(const char* txt, const Font& font, int x, int y, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);


	Image& crop(uint16_t cx, uint16_t cy, uint16_t cw, uint16_t ch);


	Image& resizeNN(uint16_t nw, uint16_t nh);


};

struct ImageInfo {
	std::string baseName;
	std::string inputPath;
};


struct Font {
	SFT sft = {nullptr, 12, 12, 0, 0, SFT_DOWNWARD_Y|SFT_RENDER_IMAGE};
	Font(const char* fontfile, uint16_t size) {
		if((sft.font = sft_loadfile(fontfile)) == NULL) {
			printf("\e[31m[ERROR] Failed to load %s\e[0m\n", fontfile);
			return;
		}
		setSize(size);
	}
	~Font() {
		sft_freefont(sft.font);
	}
	void setSize(const uint16_t size) {
		sft.xScale = size;
		sft.yScale = size;
	}
};

inline ImageInfo extractImageInfo(const std::string& inputFile) {

	const size_t dotPos   = inputFile.find_last_of('.');
	const size_t slashPos = inputFile.find_last_of('/');

	const std::string folderName = inputFile.substr(0, slashPos);

	const std::string baseName =
	inputFile.substr(slashPos + 1, dotPos - (slashPos + 1));

	const std::string inputPath = "Input/" + inputFile;

	return {baseName, inputPath};
}

#endif // IMAGE_H
