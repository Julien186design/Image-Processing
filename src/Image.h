#include <stdint.h>
#include <cstdio>
#include <complex>
#include <functional>
#include <vector>
#include <cstdint>


//legacy feature of C
#undef __STRICT_ANSI__
#define _USE_MATH_DEFINES
#include <cmath>
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

/*
std::vector<std::vector<uint8_t>> complexColors = {
	{191, 255},
	{63, 0}
};
*/

struct Image {
	uint8_t* data = NULL;
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
	ChannelIndices get_channel_indices(uint8_t r, uint8_t g, uint8_t b) {
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

	ImageType get_file_type(const char* filename);

	Image& std_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& std_convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
	Image& std_convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);

	Image& applyThresholdTransformationRegionFraction(
		int threshold,
		int fraction,
		const std::vector<int>& rectanglesToModify,
		std::function<bool(int)> condition,
		std::function<uint8_t()> transformation
	);

	static inline bool approx_equal(uint8_t a, uint8_t b, uint8_t tol) {
		return std::abs(static_cast<int>(a) - static_cast<int>(b)) <= tol;
	}

	static inline uint8_t avg_u8_round(uint8_t a, uint8_t b) {
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


	Image& diffmap(Image& img);
	Image& diffmap_scale(Image& img, uint8_t scl = 0);


	Image& grayscale_avg();

	Image& applyThresholdTransformationRegionFraction(int threshold, float fraction, std::function<bool(int)> condition, std::function<uint8_t()> transformation);


	Image& darkenBelowThreshold_ColorNuance(int threshold, int cn);
	Image& whitenBelowThreshold_ColorNuance(int threshold, int cn);
	Image& darkenAboveThreshold_ColorNuance(int threshold, int cn);
	Image& whitenAboveThreshold_ColorNuance(int threshold, int cn);


	Image& reverseBelowThreshold(int s);
	Image& reverseAboveThreshold(int s);

	Image& alternatelyDarkenAndWhitenBelowTheThreshold(int s, int first_threshold,	int last_threshold);
	Image& alternatelyDarkenAndWhitenAboveTheThreshold(int s, int first_threshold,	int last_threshold);

	Image& reversed_black_and_white(int s);
	Image& original_black_and_white(int s);

	Image& simplify_to_dominant_color_combinations(int tolerance, bool average);

	Image& darkenBelowThresholdRegionFraction(int s, int fraction, const std::vector<int>& rectanglesToModify);
	Image& whitenBelowThresholdRegionFraction(int s, int fraction, const std::vector<int>& rectanglesToModify);
	Image& darkenAboveThresholdRegionFraction(int s, int fraction, const std::vector<int>& rectanglesToModify);
	Image& whitenAboveThresholdRegionFraction(int s, int fraction, const std::vector<int>& rectanglesToModify);


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


struct Font {
	SFT sft = {NULL, 12, 12, 0, 0, SFT_DOWNWARD_Y|SFT_RENDER_IMAGE};
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
	void setSize(uint16_t size) {
		sft.xScale = size;
		sft.yScale = size;
	}
};
