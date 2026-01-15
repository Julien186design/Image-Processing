#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define BYTE_BOUND(value) value < 0 ? 0 : (value > 255 ? 255 : value)

/*
 *CLion on Linux Mint
 *Run/Edit Configurations
 *Working directory : $PROJECT8DIR$ | /[...]/CLion Projects/Image Processing


 *Dev-C++ on Windows
 *#include "stb_image.h"
 *#include "stb_image_write.h"
*/
#include <stb_image.h>
#include <stb_image_write.h>
#include <immintrin.h>
#include <cstdint>
#include <cstring>


#include "Image.h"


Image::Image(const char* filename, int channel_force) {
	if(read(filename, channel_force)) {
		printf("Read %s\n", filename);
		size = w*h*channels;
	} else {
		printf("Failed to read %s\n", filename);
	}
}

Image::Image(int w, int h, int channels) : w(w), h(h), channels(channels) {
	size = w*h*channels;
	data = new uint8_t[size];
}

Image::Image(const Image& img) : Image(img.w, img.h, img.channels) {
	memcpy(data, img.data, size);
}

Image::~Image() {
	// stbi_image_free(data);
	if (data != nullptr) {
		delete[] data;
	}
}

bool Image::read(const char* filename, int channel_force) {
	data = stbi_load(filename, &w, &h, &channels, channel_force);
	channels = channel_force == 0 ? channels : channel_force;
	return data != NULL;
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
	if(success != 0) {
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

Image& Image::darkenBelowThreshold_ColorNuance(const int threshold, const int cn) {  //the blacked pixels become black
	for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb < threshold) {
			data[i] = data[i + 1] = data[i + 2] = cn;    // used to be memset(data+i, 0, 3);
		}
	}
	return *this;
}

Image& Image::whitenBelowThreshold_ColorNuance(const int threshold, const int cn) {	//the blackest pixels become white
	const int newColor = 255 - cn;
	for(size_t i = 0; i < size; i += channels) {
		int rgb = (data[i] + data[i + 1] + data[i + 2]);
		if (rgb < threshold) {
			data[i] = data[i + 1] = data[i + 2] = newColor;
		}
	}
	return *this;
}

Image& Image::darkenAboveThreshold_ColorNuance(const int threshold, const int cn) {  //the whitest pixels become black
	for(size_t i = 0; i < size; i+=channels) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb > threshold) {
			data[i] = data[i + 1] = data[i + 2] = cn;
		}
	}
	return *this;
}

Image& Image::whitenAboveThreshold_ColorNuance(const int threshold, const int cn) {    //the whitest pixels become white
	const int newColor = 255 - cn;
	for(size_t i = 0; i < size; i+=channels) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb > threshold) {
			data[i] = data[i + 1] = data[i + 2] = newColor;
		}
	}
	return *this;
}


Image& Image::reverseAboveThreshold(int s) {
	const int threshold3 = 3 * s;
	for(size_t i = 0; i < size; i += channels) {
		int rgb = (data[i] + data[i + 1] + data[i + 2]);
		if (rgb < threshold3) {
			data[i] = 255 - data[i];
			data[i + 1] = 255 - data[i + 1];
			data[i + 2] = 255 - data[i + 2];
		}
	}
	return *this;
}

Image& Image::reverseBelowThreshold(int s) {
	const int threshold3 = 3 * s;
	for(size_t i = 0; i < size; i+=channels) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb > threshold3) {
			data[i] = 255 - data[i];
			data[i + 1] = 255 - data[i + 1];
			data[i + 2] = 255 - data[i + 2];
		}
	}
	return *this;
}

Image& Image::original_black_and_white(int s) {
	const int threshold3 = 3 * s;
	for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
		int rgb = (data[i] + data[i+1] + data[i+2]);
		if (rgb > threshold3) {
			data[i] = data[i + 1] = data[i + 2] = 255;
		} else {
			data[i] = data[i + 1] = data[i + 2] = 0;
		}
	}
	return *this;
}

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

Image& Image::reversed_black_and_white(int s) {
	const int threshold3 = 3 * s;
	for (size_t i = 0; i < size; i += channels) {

		int rgb = data[i] + data[i + 1] + data[i + 2];

		uint8_t value = (rgb < threshold3) ? 255 : 0;

		data[i] = data[i + 1] = data[i + 2] = value;
	}
	return *this;
}



Image& Image::simplify_to_dominant_color_combinations(int tolerance, bool average) {

	for (size_t i = 0; i < size; i += channels) {
		uint8_t r = data[i];
		uint8_t g = data[i + 1];
		uint8_t b = data[i + 2];

		uint8_t r_one_third_avg = avg_u8_round(r, SimpleColors::ONE_THIRD);
		uint8_t g_one_third_avg = avg_u8_round(g, SimpleColors::ONE_THIRD);
		uint8_t b_one_third_avg = avg_u8_round(b, SimpleColors::ONE_THIRD);
		
		uint8_t r_half_avg = avg_u8_round(r, SimpleColors::HALF);
		uint8_t g_half_avg = avg_u8_round(g, SimpleColors::HALF);
		uint8_t b_half_avg = avg_u8_round(b, SimpleColors::HALF);

		uint8_t r_full_avg = avg_u8_round(r, SimpleColors::FULL);
		uint8_t g_full_avg = avg_u8_round(g, SimpleColors::FULL);
		uint8_t b_full_avg = avg_u8_round(b, SimpleColors::FULL);

		std::vector<std::vector<uint8_t>> half_full_average = {
			{r_half_avg, r_full_avg},
			{g_half_avg, g_full_avg},
			{b_half_avg, b_full_avg},
			{SimpleColors::HALF, SimpleColors::FULL}
		};

		bool r_eq_g = approx_equal(r, g, tolerance);
		bool r_eq_b = approx_equal(r, b, tolerance);
		bool g_eq_b = approx_equal(g, b, tolerance);

		// Case 1: All three color channels are equal
		if (r_eq_g && r_eq_b) {
			if (average) {
				data[i] = r_one_third_avg;
				data[i + 1] = g_one_third_avg;
				data[i + 2] = b_one_third_avg;
			} else {
				data[i] = data[i + 1] = data[i + 2] = SimpleColors::ONE_THIRD;
			}
			continue;
		}

		// Case 2: Two color channels are equal and greater than the third
		if (r_eq_g) {
			if (r > b + tolerance) {
				data[i] = data[i + 1] = SimpleColors::HALF; data[i + 2] = 0;
			} else {
				data[i] = data[i + 1] = 0; data[i + 2] = SimpleColors::HALF;
			}
		} else if (r_eq_b) {
			if (r > g + tolerance) {
				data[i] = data[i + 2] = SimpleColors::HALF; data[i + 1] = 0;
			} else {
				data[i] = data[i + 2] = 0; data[i + 1] = SimpleColors::HALF;
			}
		} else if (g_eq_b) {
			if (g > r + tolerance) {
				data[i + 1] = data[i + 2] = SimpleColors::HALF; data[i] = 0;
			} else {
				data[i + 1] = data[i + 2] = 0; data[i] = SimpleColors::HALF;
			}
		}
		// Case 3: All three color channels are distinct, in descending order
		else {
			ChannelIndices indices = get_channel_indices(r, g, b);
			if (average) {
				data[i + indices.max] = half_full_average[indices.max][1];
				data[i + indices.mid] = half_full_average[indices.mid][0];
				data[i + indices.min] = 0;
			} else {
				data[i + indices.max] = SimpleColors::FULL;
				data[i + indices.mid] = SimpleColors::HALF;
				data[i + indices.min] = 0;
			}
		}
	}
	return *this;
}


//fraction by rectangles

Image& Image::applyThresholdTransformationRegionFraction(
	int threshold,
	int fraction,
	const std::vector<int>& rectanglesToModify,
	std::function<bool(int)> condition,
	std::function<uint8_t()> transformation
) {
	int numRectanglesPerRow = fraction * 2;  // Nombre de rectangles par ligne
	int numRectanglesPerCol = fraction * 2;  // Nombre de rectangles par colonne
	int totalRectangles = numRectanglesPerRow * numRectanglesPerCol;

	int rectWidth = w / numRectanglesPerRow;  // Largeur de chaque rectangle
	int rectHeight = h / numRectanglesPerCol; // Hauteur de chaque rectangle

	for (int rectIndex : rectanglesToModify) {
		if (rectIndex < 0 || rectIndex >= totalRectangles) {
			continue; // Ignorer les indices invalides
		}

		int rectRow = rectIndex / numRectanglesPerRow;
		int rectCol = rectIndex % numRectanglesPerRow;

		int startX = rectCol * rectWidth;
		int startY = rectRow * rectHeight;
		int endX = (rectCol + 1) * rectWidth;
		int endY = (rectRow + 1) * rectHeight;

		// Assurez-vous de ne pas dépasser les limites de l'image
		endX = std::min(endX, w);
		endY = std::min(endY, h);

		for (int y = startY; y < endY; ++y) {
			for (int x = startX; x < endX; ++x) {
				size_t i = (y * w + x) * channels;
				int rgb = (data[i] + data[i + 1] + data[i + 2]);

				if (condition(rgb)) {
					uint8_t value = transformation();
					data[i] = data[i + 1] = data[i + 2] = value;
				}
			}
		}
	}
	return *this;
}

Image& Image::darkenBelowThresholdRegionFraction(int s, int fraction, const std::vector<int>& rectanglesToModify) {
	return applyThresholdTransformationRegionFraction(
		3 * s,
		fraction,
		rectanglesToModify,
		[s](int rgb) { return rgb < 3 * s; },
		[]() { return 0; }
	);
}

Image& Image::whitenBelowThresholdRegionFraction(int s, int fraction, const std::vector<int>& rectanglesToModify) {
	return applyThresholdTransformationRegionFraction(
		3 * s,
		fraction,
		rectanglesToModify,
		[s](int rgb) { return rgb < 3 * s; },
		[]() { return 255; }
	);
}

Image& Image::darkenAboveThresholdRegionFraction(int s, int fraction, const std::vector<int>& rectanglesToModify) {
	return applyThresholdTransformationRegionFraction(
		3 * s,
		fraction,
		rectanglesToModify,
		[s](int rgb) { return rgb > 3 * s; },
		[]() { return 0; }
	);
}

Image& Image::whitenAboveThresholdRegionFraction(int s, int fraction, const std::vector<int>& rectanglesToModify) {
	return applyThresholdTransformationRegionFraction(
		3 * s,
		fraction,
		rectanglesToModify,
		[s](int rgb) { return rgb > 3 * s; },
		[]() { return 255; }
	);
}

Image& Image::operator=(const Image& img) {
	if (this == &img) {
		return *this;  // Protection auto-affectation
	}

	// Libération de l'ancien buffer
	if (data != nullptr) {
		delete[] data;
	}

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

Image& Image::std_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	uint8_t new_data[w*h];
	uint64_t center = cr*ker_w + cc;
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for(long i = -((long)cr); i<(long)ker_h-cr; ++i) {
			long row = ((long)k/channels)/w-i;
			if(row < 0 || row > h-1) {
				continue;
			}
			for(long j = -((long)cc); j<(long)ker_w-cc; ++j) {
				long col = ((long)k/channels)%w-j;
				if(col < 0 || col > w-1) {
					continue;
				}
				c += ker[center+i*(long)ker_w+j]*data[(row*w+col)*channels+channel];
			}
		}
		new_data[k/channels] = (uint8_t)BYTE_BOUND(round(c));
	}
	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}


Image& Image::std_convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	uint8_t new_data[w*h];
	uint64_t center = cr*ker_w + cc;
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for(long i = -((long)cr); i<(long)ker_h-cr; ++i) {
			long row = ((long)k/channels)/w-i;
			if(row < 0) {
				row = 0;
			} else if(row > h-1) {
				row = h-1;
			}
			for(long j = -((long)cc); j<(long)ker_w-cc; ++j) {
				long col = ((long)k/channels)%w-j;
				if(col < 0) {
					col = 0;
				} else if(col > w-1) {
					col = w-1;
				}
				c += ker[center+i*(long)ker_w+j]*data[(row*w+col)*channels+channel];
			}
		}
		new_data[k/channels] = (uint8_t)BYTE_BOUND(round(c));
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
		for(long i = -((long)cr); i<(long)ker_h-cr; ++i) {
			long row = ((long)k/channels)/w-i;
			if(row < 0) {
				row = row%h + h;
			} else if(row > h-1) {
				row %= h;
			}
			for(long j = -((long)cc); j<(long)ker_w-cc; ++j) {
				long col = ((long)k/channels)%w-j;
				if(col < 0) {
					col = col%w + w;
				} else if(col > w-1) {
					col %= w;
				}
				c += ker[center+i*(long)ker_w+j]*data[(row*w+col)*channels+channel];
			}
		}
		new_data[k/channels] = (uint8_t)BYTE_BOUND(round(c));
	}
	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}






uint32_t Image::rev(uint32_t n, uint32_t a) {
	uint8_t max_bits = (uint8_t)ceil(log2(n));
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
	//x in standard order
	if(x != X) {
		memcpy(X, x, n*sizeof(std::complex<double>));
	}

	//Gentleman-Sande butterfly
	uint32_t sub_probs = 1;
	uint32_t sub_prob_size = n;
	uint32_t half;
	uint32_t i;
	uint32_t j_begin;
	uint32_t j_end;
	uint32_t j;
	std::complex<double> w_step;
	std::complex<double> w;
	std::complex<double> tmp1, tmp2;
	while(sub_prob_size>1) {
		half = sub_prob_size>>1;
		w_step = std::complex<double>(cos(-2*M_PI/sub_prob_size), sin(-2*M_PI/sub_prob_size));
		for(i=0; i<sub_probs; ++i) {
			j_begin = i*sub_prob_size;
			j_end = j_begin+half;
			w = std::complex<double>(1,0);
			for(j=j_begin; j<j_end; ++j) {
				tmp1 = X[j];
				tmp2 = X[j+half];
				X[j] = tmp1+tmp2;
				X[j+half] = (tmp1-tmp2)*w;
				w *= w_step;
			}
		}
		sub_probs <<= 1;
		sub_prob_size = half;
	}
	//X in bit reversed order
}

void Image::ifft(uint32_t n, std::complex<double> X[], std::complex<double>* x) {
	//X in bit reversed order
	if(X != x) {
		memcpy(x, X, n*sizeof(std::complex<double>));
	}

	//Cooley-Tukey butterfly
	uint32_t sub_probs = n>>1;
	uint32_t sub_prob_size;
	uint32_t half = 1;
	uint32_t i;
	uint32_t j_begin;
	uint32_t j_end;
	uint32_t j;
	std::complex<double> w_step;
	std::complex<double> w;
	std::complex<double> tmp1, tmp2;
	while(half<n) {
		sub_prob_size = half<<1;
		w_step = std::complex<double>(cos(2*M_PI/sub_prob_size), sin(2*M_PI/sub_prob_size));
		for(i=0; i<sub_probs; ++i) {
			j_begin = i*sub_prob_size;
			j_end = j_begin+half;
			w = std::complex<double>(1,0);
			for(j=j_begin; j<j_end; ++j) {
				tmp1 = x[j];
				tmp2 = w*x[j+half];
				x[j] = tmp1+tmp2;
				x[j+half] = tmp1-tmp2;
				w *= w_step;
			}
		}
		sub_probs >>= 1;
		half = sub_prob_size;
	}
	for(uint32_t i=0; i<n; ++i) {
		x[i] /= n;
	}
	//x in standard order
}

void Image::dft_2D(uint32_t m, uint32_t n, std::complex<double> x[], std::complex<double>* X) {
	//x in row-major & standard order
	std::complex<double>* intermediate = new std::complex<double>[m*n];
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
	std::complex<double>* intermediate = new std::complex<double>[m*n];
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

void Image::pad_kernel(uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double>* pad_ker) {
	//padded so center of kernel is at top left
	for(long i=-((long)cr); i<(long)ker_h-cr; ++i) {
		uint32_t r = (i<0) ? i+ph : i;
		for(long j=-((long)cc); j<(long)ker_w-cc; ++j) {
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
	std::complex<double>* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			pad_img[i*pw+j] = std::complex<double>(data[(i*w+j)*channels+channel],0);
		}
	}

	//pad kernel
	std::complex<double>* pad_ker = new std::complex<double>[psize];
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
	uint32_t pw = 1<<((uint8_t)ceil(log2(w+ker_w-1)));
	uint32_t ph = 1<<((uint8_t)ceil(log2(h+ker_h-1)));
	uint64_t psize = pw*ph;

	//pad image
	std::complex<double>* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<ph; ++i) {
		uint32_t r = (i<h) ? i : ((i<h+cr ? h-1 : 0));
		for(uint32_t j=0; j<pw; ++j) {
			uint32_t c = (j<w) ? j : ((j<w+cc ? w-1 : 0));
			pad_img[i*pw+j] = std::complex<double>(data[(r*w+c)*channels+channel],0);
		}
	}

	//pad kernel
	std::complex<double>* pad_ker = new std::complex<double>[psize];
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
Image& Image::fd_convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	//calculate padding
	uint32_t pw = 1<<((uint8_t)ceil(log2(w+ker_w-1)));
	uint32_t ph = 1<<((uint8_t)ceil(log2(h+ker_h-1)));
	uint64_t psize = pw*ph;

	//pad image
	std::complex<double>* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<ph; ++i) {
		uint32_t r = (i<h) ? i : ((i<h+cr ? i%h : h-ph+i));
		for(uint32_t j=0; j<pw; ++j) {
			uint32_t c = (j<w) ? j : ((j<w+cc ? j%w : w-pw+j));
			pad_img[i*pw+j] = std::complex<double>(data[(r*w+c)*channels+channel],0);
		}
	}

	//pad kernel
	std::complex<double>* pad_ker = new std::complex<double>[psize];
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


Image& Image::convolve_linear(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	} else {
		return std_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	}
}
Image& Image::convolve_clamp_to_border(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_clamp_to_border(channel, ker_w, ker_h, ker, cr, cc);
	} else {
		return std_convolve_clamp_to_border(channel, ker_w, ker_h, ker, cr, cc);
	}
}
Image& Image::convolve_cyclic(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_cyclic(channel, ker_w, ker_h, ker, cr, cc);
	} else {
		return std_convolve_cyclic(channel, ker_w, ker_h, ker, cr, cc);
	}
}


Image& Image::diffmap(Image& img) {
	int compare_width = fmin(w,img.w);
	int compare_height = fmin(h,img.h);
	int compare_channels = fmin(channels,img.channels);
	for(uint32_t i=0; i<compare_height; ++i) {
		for(uint32_t j=0; j<compare_width; ++j) {
			for(uint8_t k=0; k<compare_channels; ++k) {
				data[(i*w+j)*channels+k] = BYTE_BOUND(abs(data[(i*w+j)*channels+k] - img.data[(i*img.w+j)*img.channels+k]));
			}
		}
	}
	return *this;
}

Image& Image::diffmap_scale(Image& img, uint8_t scl) {
	int compare_width = fmin(w,img.w);
	int compare_height = fmin(h,img.h);
	int compare_channels = fmin(channels,img.channels);
	uint8_t largest = 0;
	for(uint32_t i=0; i<compare_height; ++i) {
		for(uint32_t j=0; j<compare_width; ++j) {
			for(uint8_t k=0; k<compare_channels; ++k) {
				data[(i*w+j)*channels+k] = BYTE_BOUND(abs(data[(i*w+j)*channels+k] - img.data[(i*img.w+j)*img.channels+k]));
				largest = fmax(largest, data[(i*w+j)*channels+k]);
			}
		}
	}
	scl = 255/fmax(1, fmax(scl, largest));
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
			int gray = 0.2126*data[i] + 0.7152*data[i+1] + 0.0722*data[i+2];
			memset(data+i, gray, 3);
		}
	}
	return *this;
}


Image& Image::color_mask(float r, float g, float b) {
	if(channels < 3) {
		printf("\e[31m[ERROR] Color mask requires at least 3 channels, but this image has %d channels\e[0m\n", channels);
	} else {
		for(size_t i = 0; i < size; i+=static_cast<size_t>(channels)) {
			data[i] *= r;
			data[i+1] *= g;
			data[i+2] *= b;
		}
	}
	return *this;
}


Image& Image::encodeMessage(const char* message) {
	uint32_t len = strlen(message) * 8;
	if(len + STEG_HEADER_SIZE > size) {
		printf("\e[31m[ERROR] This message is too large (%lu bits / %zu bits)\e[0m\n", len+STEG_HEADER_SIZE, size);
		return *this;
	}

	for(uint8_t i = 0; i < STEG_HEADER_SIZE; ++i) {
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
	uint32_t len = 0;
	for(uint8_t i = 0; i < STEG_HEADER_SIZE; ++i) {
		len = (len << 1) | (data[i] & 1);
	}
	*messageLength = len / 8;

	for(uint32_t i = 0; i < len; ++i) {
		buffer[i/8] = (buffer[i/8] << 1) | (data[i+STEG_HEADER_SIZE] & 1);
	}


	return *this;
}




Image& Image::flipX() {
	uint8_t tmp[4];
	uint8_t* px1;
	uint8_t* px2;
	for(int y = 0; y < h; ++y) {
		for(int x = 0; x < w/2; ++x) {
			px1 = &data[(x + y * w) * channels];
			px2 = &data[((w - 1 - x) + y * w) * channels];

			memcpy(tmp, px1, channels);
			memcpy(px1, px2, channels);
			memcpy(px2, tmp, channels);
		}
	}
	return *this;
}
Image& Image::flipY() {
	uint8_t tmp[4];
	uint8_t* px1;
	uint8_t* px2;
	for(int x = 0; x < w; ++x) {
		for(int y = 0; y < h/2; ++y) {
			px1 = &data[(x + y * w) * channels];
			px2 = &data[(x + (h - 1 - y) * w) * channels];

			memcpy(tmp, px1, channels);
			memcpy(px1, px2, channels);
			memcpy(px2, tmp, channels);
		}
	}
	return *this;
}




Image& Image::overlay(const Image& source, int x, int y) {

	uint8_t* srcPx;
	uint8_t* dstPx;

	for(int sy = 0; sy < source.h; ++sy) {
		if(sy + y < 0) {
			continue;
		} else if(sy + y >= h) {
			break;
		}
		for(int sx = 0; sx < source.w; ++sx) {
			if(sx + x < 0) {
				continue;
			} else if(sx + x >= w) {
				break;
			}
			srcPx = &source.data[(sx + sy * source.w) * source.channels];
			dstPx = &data[(sx + x + (sy + y) * w) * channels];

			float srcAlpha = source.channels < 4 ? 1 : srcPx[3] / 255.f;
			float dstAlpha = channels < 4 ? 1 : dstPx[3] / 255.f;

			if(srcAlpha > .99 && dstAlpha > .99) {
				if(source.channels >= channels) {
					memcpy(dstPx, srcPx, channels);
				} else {
					// In case our source image is grayscale and the dest one isnt
					memset(dstPx, srcPx[0], channels);
				}
			} else {
				float outAlpha = srcAlpha + dstAlpha * (1 - srcAlpha);
				if(outAlpha < .01) {
					memset(dstPx, 0, channels);
				} else {
					for(int chnl = 0; chnl < channels; ++chnl) {
						dstPx[chnl] = (uint8_t)BYTE_BOUND((srcPx[chnl]/255.f * srcAlpha + dstPx[chnl]/255.f * dstAlpha * (1 - srcAlpha)) / outAlpha * 255.f);
					}
					if(channels > 3) {
						dstPx[3] = (uint8_t)BYTE_BOUND(outAlpha * 255.f);
					}
				}
			}

		}

	}
	return *this;
}



Image& Image::crop(uint16_t cx, uint16_t cy, uint16_t cw, uint16_t ch) {
	size = cw * ch * channels;
	uint8_t* croppedImage = new uint8_t[size];
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




Image& Image::resizeNN(uint16_t nw, uint16_t nh) {
	size = nw * nh * channels;
	uint8_t* newImage = new uint8_t[size];

	float scaleX = (float)nw / (w);
	float scaleY = (float)nh / (h);
	uint16_t sx, sy;

	for(uint16_t y = 0; y < nh; ++y) {
		sy = (uint16_t)(y / scaleY);
		for(uint16_t x = 0; x < nw; ++x) {
			sx = (uint16_t)(x / scaleX);

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
