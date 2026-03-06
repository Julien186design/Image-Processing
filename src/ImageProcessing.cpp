#include "ImageProcessing.h"

EdgeDetectorResult process_edge_detection_core(
    const uint8_t* grayData,
    const int width,
    const int height,
    const double threshold
) {
	const size_t imgSize = width * height;

	// Determine the number of threads to use
	const int max_threads = omp_get_max_threads();
	const int threads_to_use = std::max(1, max_threads - THREAD_OFFSET);
	omp_set_num_threads(threads_to_use);

	// Pre-computed Gaussian kernel
	constexpr double inv16 = 1.0 / 16.0;
	constexpr double gauss[9] = {
		inv16, 2 * inv16, inv16,
		2 * inv16, 4 * inv16, 2 * inv16,
		inv16, 2 * inv16, inv16
	};

	std::vector blurData(imgSize, 0.0);
	std::vector<double> tx(imgSize), ty(imgSize);
	std::vector<double> gx(imgSize), gy(imgSize);
	std::vector<double> g(imgSize), theta(imgSize);

	// === Gaussian blur ===
#pragma omp parallel for default(none) shared(blurData, grayData, width, height, gauss)
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
#pragma omp parallel for default(none) shared(tx, ty, blurData, width, height)
	for (int r = 0; r < height; ++r) {
		for (int c = 1; c < width - 1; ++c) {
			const size_t idx = r * width + c;
			tx[idx] = blurData[idx + 1] - blurData[idx - 1];
			ty[idx] = 47 * blurData[idx + 1] + 162 * blurData[idx] + 47 * blurData[idx - 1];
		}
	}

#pragma omp parallel for default(none) shared(gx, gy, tx, ty, width, height)
	for (int c = 1; c < width - 1; ++c) {
		for (int r = 1; r < height - 1; ++r) {
			const size_t idx = r * width + c;
			gx[idx] = 47 * tx[idx + width] + 162 * tx[idx] + 47 * tx[idx - width];
			gy[idx] = ty[idx + width] - ty[idx - width];
		}
	}

	// === Magnitude/angle computation + min/max reduction ===
	double mx = -INFINITY, mn = INFINITY;
#pragma omp parallel default(none) shared(g, gx, gy, theta, mx, mn, imgSize)
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

	// === HSL to RGB conversion ===
	std::vector<uint8_t> outputRGB(imgSize * 3);
	const double range = (mx == mn) ? 1.0 : (mx - mn);

#pragma omp parallel for default(none) shared(outputRGB, g, theta, mn, range, threshold, imgSize)
	for (int k = 0; k < imgSize; ++k) {
		const double h = theta[k] * 180.0 / M_PI + 180.0;
		const double v = ((g[k] - mn) / range > threshold) ? (g[k] - mn) / range : 0.0;
		const double s = v, l = v;

		const double c = (1 - std::abs(2 * l - 1)) * s;
		const double x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
		const double m = l - c / 2.0;

		double rt = 0, gt = 0, bt = 0;
		if (h < 60) {
			rt = c;
			gt = x;
		} else if (h < 120) {
			rt = x;
			gt = c;
		} else if (h < 180) {
			gt = c;
			bt = x;
		} else if (h < 240) {
			gt = x;
			bt = c;
		} else if (h < 300) {
			bt = c;
			rt = x;
		} else {
			bt = x;
			rt = c;
		}

		outputRGB[k * 3] = static_cast<uint8_t>(255 * (rt + m));
		outputRGB[k * 3 + 1] = static_cast<uint8_t>(255 * (gt + m));
		outputRGB[k * 3 + 2] = static_cast<uint8_t>(255 * (bt + m));
	}

	return {std::move(outputRGB), mn, mx};
}
