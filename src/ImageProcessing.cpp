#include "ImageProcessing.h"
#include "TransformationsConfig.h"
#include <vector>
#include <functional>
#include <string>
#include <cstring>
#include <immintrin.h> // For AVX2 intrinsics


// ============================================================================
// TYPES ET STRUCTURES
// ============================================================================

using GenericTransformationFunc = std::function<void(Image&, int, const std::vector<int>&)>;

struct TransformationContext {
    const Image& baseImage;
    const std::string& baseName;
    int firstThreshold;
    int lastThreshold;
    int step;
    bool saveAt120;
    std::string folder120;
};



// ============================================================================
// FONCTION GÉNÉRIQUE CENTRALE
// ============================================================================


void applyAndSaveGenericTransformations(
    const Image& baseImage,
    const std::vector<GenericTransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const std::string& transformationType,
    const int threshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120,
    const std::vector<int>& additionalParams
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int currentThreshold = threshold; currentThreshold <= lastThreshold; currentThreshold += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {
            // LOGGING
            // printf("  Transformation %zu/%zu: %s\n", i+1, transforms.size(), suffixes[i].c_str());

            modified.resetFrom(baseImage);

            transforms[i](modified.get(), currentThreshold, additionalParams);

            std::string outputPath = OutputPathBuilder::buildStandard(
                outputDirs[i],
                baseName,
                transformationType,
                suffixes[i],
                currentThreshold / 3
            );
            modified.saveAs(outputPath.c_str());

            if (saveAt120 && currentThreshold == 360) {
                std::string specialPath = OutputPathBuilder::build120(
                    folder120,
                    baseName,
                    transformationType,
                    suffixes[i]
                );
                modified.saveAs(specialPath.c_str());
            }
        }

    }
}

void applyTransformationsWithMultipleColorNuances(
    const Image& baseImage,
    const std::vector<GenericTransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const std::string& transformationType,
    const int threshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120,
    const std::vector<int>& colorNuance
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int currentThreshold = threshold; currentThreshold <= lastThreshold; currentThreshold += step) {
        for (size_t i = 0; i < transforms.size(); ++i) {
            // LOGGING
            // printf("  Transformation %zu/%zu: %s\n", i+1, transforms.size(), suffixes[i].c_str());

            modified.resetFrom(baseImage);

            transforms[i](modified.get(), currentThreshold, colorNuance);

            std::string outputPath = OutputPathBuilder::buildStandard(
                outputDirs[i],
                baseName,
                transformationType,
                suffixes[i],
                currentThreshold / 3
            );
            modified.saveAs(outputPath.c_str());

            if (saveAt120 && currentThreshold == 120) {
                std::string specialPath = OutputPathBuilder::build120(
                    folder120,
                    baseName,
                    transformationType,
                    suffixes[i]
                );
                modified.saveAs(specialPath.c_str());
            }
        }

    }
}

// ============================================================================
// FONCTIONS DE COMPATIBILITÉ (LEGACY)
// ============================================================================

void applyTransformationsWrapper(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const int threshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120
) {
    applyAndSaveGenericTransformations(
        baseImage,
        wrapSimpleTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " - ",
        threshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        {}
    );
}

void applyAndSaveTransformations(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const int threshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120
) {
    applyTransformationsWrapper(
        baseImage,
        transforms,
        outputDirs,
        suffixes,
        baseName,
        threshold,
        lastThreshold,
        step,
        saveAt120,
        folder120
    );
}

void applyAndSaveReversalTransformations(
    const Image& baseImage,
    const std::vector<TransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const int threshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120
) {
    applyAndSaveGenericTransformations(
        baseImage,
        wrapSimpleTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " Reversal - ",
        threshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        {}
    );
}

void applyAndSaveAlternatingTransformations(
    const Image& baseImage,
    const std::vector<AlternatingTransformation>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const int firstThreshold,
    const int lastThreshold,
    const int step,
    const bool saveAt120,
    const std::string& folder120
) {
    const std::vector<int> additionalParams = {firstThreshold, lastThreshold, step};

    applyAndSaveGenericTransformations(
        baseImage,
        wrapAlternatingTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " Alternating - ",
        firstThreshold,
        lastThreshold,
        step,
        saveAt120,
        folder120,
        additionalParams
    );
}

void applyAndSavePartialTransformations(
    const Image& baseImage,
    const std::vector<PartialTransformationFunc>& transforms,
    const std::vector<std::string>& outputDirs,
    const std::vector<std::string>& suffixes,
    const std::string& baseName,
    const int threshold,
    const int lastThreshold,
    const int step,
    const int fraction,
    const std::vector<int>& rectanglesToModify
) {
    std::vector<int> partialParams = {fraction};
    partialParams.insert(partialParams.end(), rectanglesToModify.begin(), rectanglesToModify.end());

    applyAndSaveGenericTransformations(
        baseImage,
        wrapPartialTransforms(transforms),
        outputDirs,
        suffixes,
        baseName,
        " - ",
        threshold,
        lastThreshold,
        step,
        false,
        "",
        partialParams
    );
}

// ============================================================================
// TRANSFORMATIONS SPÉCIALISÉES
// ============================================================================

void oneColorTransformations(
    const Image& baseImage,
    const std::string& baseName,
    const std::vector<int>& tolerance
) {
    const int start = tolerance[0];
    const int end   = tolerance[1];
    const int step  = tolerance[2];
    bool average = false;

	const std::string ONE_COLOR_FOLDER = std::string(OUTPUT_FOLDER) + "One Color/";
	const std::string DIFFMAP_FOLDER = std::string(OUTPUT_FOLDER) + "Diffmap/";

	ImageBuffer buffer(baseImage.w, baseImage.h, baseImage.channels);

    for (int t = start; t <= end; t += step) {
        for (int i = 0; i < 2; ++i) {
        	buffer.resetFrom(baseImage);

        	buffer.get().simplify_to_dominant_color_combinations(t, average);

            std::string outputPath = ONE_COLOR_FOLDER + baseName + " - Average " + std::to_string(average) +
                                    " - Tolerance " + std::to_string(t) + ".png";
        	buffer.saveAs(outputPath.c_str());

            average = !average;
        }
    }

	for (int i = start; i <= end; i += step) {
		std::string path1 = ONE_COLOR_FOLDER + baseName + " - Average 1 - Tolerance " + std::to_string(i) + ".png";
		std::string path2 = ONE_COLOR_FOLDER + baseName + " - Average 0 - Tolerance " + std::to_string(i) + ".png";

		Image image1(path1.c_str());
		Image image2(path2.c_str());

		ImageBuffer diffBuffer(image1.w, image1.h, image1.channels);
		diffBuffer.resetFrom(image1);
		diffBuffer.get().diffmap(image2);

		std::string outputPath = DIFFMAP_FOLDER + baseName + " - Diffmap - Tolerance " + std::to_string(i) + ".png";
		diffBuffer.saveAs(outputPath.c_str());
	}
}

void several_colors_transformations(
    const Image& baseImage,
    const std::string& baseName,
    const int firstThreshold,
    const int lastThreshold,
    const int step,
    const int firstColorNuance,
    const int lastColorNuance,
    const int stepColorNuance
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    for (int currentThreshold = firstThreshold; currentThreshold <= lastThreshold; currentThreshold += step) {

        for (size_t transformIdx = 0; transformIdx < colors_nuances_transformations.size(); ++transformIdx) {
            for (int currentColorNuance = firstColorNuance;
                currentColorNuance <= lastColorNuance; currentColorNuance += stepColorNuance) {

                modified.resetFrom(baseImage);

                colors_nuances_transformations[transformIdx](modified.get(), currentThreshold, currentColorNuance);

                std::string outputPath = OutputPathBuilder::buildStandard(
                    total_step_by_step_output_dirs[transformIdx],
                    baseName,
                    " - ",
                    total_step_by_step_suffixes[transformIdx],
                    currentThreshold / 3
                );

                outputPath += " - CN " + std::to_string(currentColorNuance) + ".png";

                modified.saveAs(outputPath.c_str());
            }
        }
    }
}



void several_colors_partial_transformations(
    const Image& baseImage,
    const std::vector<GenericTransformationFuncWithColorNuances>& transforms,
    const std::string& baseName,
    const int firstThreshold,
    const int lastThreshold,
    const int step,
    const int fraction,
    std::vector<int>& rectangles,
    const int firstColorNuance,
    const int lastColorNuance,
    const int stepColorNuance
) {
    ImageBuffer modified(baseImage.w, baseImage.h, baseImage.channels);

    const std::vector<std::string> partialDirs = generatePartialOutputDirs();
    const std::vector<std::string> partialSuffixes = generatePartialSuffixes();

    bool diagonal = false;

    if (rectangles[0] == -1) {
        rectangles.erase(rectangles.begin());
        diagonal = true;
    }

    for (int currentThreshold = firstThreshold; currentThreshold <= lastThreshold; currentThreshold += step) {
        for (size_t transformIdx = 0; transformIdx < partialTransformationsFunc.size(); ++transformIdx) {
            for (int currentColorNuance = firstColorNuance;
                currentColorNuance <= lastColorNuance; currentColorNuance += stepColorNuance) {

                modified.resetFrom(baseImage);

                transforms[transformIdx](modified.get(), currentThreshold,
                    currentColorNuance, fraction, rectangles);

                std::string outputPath = OutputPathBuilder::buildStandard(
                    partialDirs[transformIdx],
                    baseName,
                    " - ",
                    partialSuffixes[transformIdx],
                    currentThreshold / 3
                );

                if (diagonal) {
                    outputPath += " - " + std::to_string(rectangles.size()) + " Squares ";
                } else {
                    outputPath += " - Rectangles " + std::to_string(rectangles[0]) +
                    " - " + std::to_string(rectangles.back());
                }
                outputPath += " CN " +
                        std::to_string(currentColorNuance) + ".png";
                modified.saveAs(outputPath.c_str());
            }
        }
    }
}



void edge_detector(
	const Image& baseImage,
	const std::string& baseName
) {
	std::string folderEdgeDetector = "Edge Detector/";
	Image img = baseImage;
	img.grayscale_avg();
	int img_size = img.w*img.h;

	Image gray_img(img.w, img.h, 1);
	printf("Dimensions de gray_img : %d x %d, data=%p\n", gray_img.w, gray_img.h, gray_img.data);
	if (gray_img.w <= 0 || gray_img.h <= 0 || gray_img.data == nullptr) {
		fprintf(stderr, "Erreur : gray_img invalide !\n");
		return;
	}

	for(uint64_t k=0; k<img_size; ++k) {
		gray_img.data[k] = img.data[img.channels*k];
	}

	std::string outputPath = std::string(OUTPUT_FOLDER) + folderEdgeDetector + baseName + " gray.png";
	// gray_img.write(outputPath.c_str());

	// blur
	Image blur_img(img.w, img.h, 1);
	constexpr double inv16 = 1.0 / 16.0;
	double gauss[9] = {
		inv16, 2*inv16, inv16,
		2*inv16, 4*inv16, 2*inv16,
		inv16, 2*inv16, inv16
	};
	printf("Noyau (kernel) : [");
	for (int i = 0; i < 9; ++i) printf("%.2f, ", gauss[i]);
	printf("]\n");

	gray_img.convolve_linear(0, 3, 3, gauss, 1, 1);
	for(uint64_t k=0; k<img_size; ++k) {
		blur_img.data[k] = gray_img.data[k];
	}
	/*
	outputPath = std::string(OUTPUT_FOLDER) + folderEdgeDetector + baseName + " blur.png";
	blur_img.write(outputPath.c_str());
	*/

	// edge detection
	auto tx = new double[img_size]();
	auto ty = new double[img_size]();
	auto gx = new double[img_size]();
	auto gy = new double[img_size]();


	//seperable convolution

	// Première boucle : remplir tx/ty pour r ∈ [0, h-1]
	for(uint32_t r=0; r<blur_img.h; ++r) {
		for(uint32_t c=1; c<blur_img.w-1; ++c) {
			tx[r*blur_img.w+c] = blur_img.data[r*blur_img.w+c+1] - blur_img.data[r*blur_img.w+c-1];
			ty[r*blur_img.w+c] = 47*blur_img.data[r*blur_img.w+c+1] + 162*blur_img.data[r*blur_img.w+c] + 47*blur_img.data[r*blur_img.w+c-1];
		}
	}
	for(uint32_t c=1; c<blur_img.w-1; ++c) {
		for(uint32_t r=1; r<blur_img.h-1; ++r) {
			gx[r*blur_img.w+c] = 47*tx[(r+1)*blur_img.w+c] + 162*tx[r*blur_img.w+c] + 47*tx[(r-1)*blur_img.w+c];
			gy[r*blur_img.w+c] = ty[(r+1)*blur_img.w+c] - ty[(r-1)*blur_img.w+c];
		}
	}

	delete[] tx;
	delete[] ty;

	//make test images
	double mxx = -INFINITY,
		mxy = -INFINITY,
		mnx = INFINITY,
		mny = INFINITY;
	for(uint64_t k=0; k<img_size; ++k) {
		mxx = fmax(mxx, gx[k]);
		mxy = fmax(mxy, gy[k]);
		mnx = fmin(mnx, gx[k]);
		mny = fmin(mny, gy[k]);
	}
	Image Gx(img.w, img.h, 1);
	Image Gy(img.w, img.h, 1);
	for(uint64_t k=0; k<img_size; ++k) {
		Gx.data[k] = static_cast<uint8_t>(255*(gx[k]-mnx)/(mxx-mnx));
		Gy.data[k] = static_cast<uint8_t>(255*(gy[k]-mny)/(mxy-mny));
	}
	/*
	outputPath = std::string(OUTPUT_FOLDER) + folderEdgeDetector + baseName + " GX.png";
	Gx.write(outputPath.c_str());
	outputPath = std::string(OUTPUT_FOLDER) + folderEdgeDetector + baseName + " Gy.png";
	Gy.write(outputPath.c_str());
	*/
	// fun part
	double threshold = 0.09;
	auto g = new double[img_size];
	auto theta = new double[img_size];
	double x, y;
	for(uint64_t k=0; k<img_size; ++k) {
		x = gx[k];
		y = gy[k];
		g[k] = sqrt(x*x + y*y);
		theta[k] = atan2(y, x);
	}


	//make images
	double mx = -INFINITY,
		mn = INFINITY;
	for(uint64_t k=0; k<img_size; ++k) {
		mx = fmax(mx, g[k]);
		mn = fmin(mn, g[k]);
	}
	Image G(img.w, img.h, 1);
	Image GT(img.w, img.h, 3);

	double h, s, l;
	double v;
	for(uint64_t k=0; k<img_size; ++k) {
		//theta to determine hue
		h = theta[k]*180./M_PI + 180.;

		//v is the relative edge strength
		if(mx == mn) {
			v = 0;
		}
		else {
			v = (g[k]-mn)/(mx-mn) > threshold ? (g[k]-mn)/(mx-mn) : 0;
		}
		s = l = v;

		//hsl => rgb
		double c = (1-abs(2*l-1))*s;
		double x = c*(1-abs(fmod((h/60),2)-1));
		double m = l-c/2;

		double rt, gt, bt;
		rt=bt=gt = 0;
		if(h < 60) {
			rt = c;
			gt = x;
		}
		else if(h < 120) {
			rt = x;
			gt = c;
		}
		else if(h < 180) {
			gt = c;
			bt = x;
		}
		else if(h < 240) {
			gt = x;
			bt = c;
		}
		else if(h < 300) {
			bt = c;
			rt = x;
		}
		else {
			bt = x;
			rt = c;
		}

		uint8_t red, green, blue;
		red = static_cast<uint8_t>(255*(rt+m));
		green = static_cast<uint8_t>(255*(gt+m));
		blue = static_cast<uint8_t>(255*(bt+m));

		GT.data[k*3] = red;
		GT.data[k*3+1] = green;
		GT.data[k*3+2] = blue;

		G.data[k] = static_cast<uint8_t>(255 * v);
	}
	/*
	outputPath = std::string(OUTPUT_FOLDER) + folderEdgeDetector + baseName + " G.png";
	//G.write(outputPath.c_str());
	*/
	outputPath = std::string(OUTPUT_FOLDER) + folderEdgeDetector + baseName + " GT.png";
	GT.write(outputPath.c_str());

	delete[] gx;
	delete[] gy;
	delete[] g;
	delete[] theta;
}


// ============================================================================
// PIPELINE PRINCIPAL
// ============================================================================


void processImageTransforms(
    const std::string& inputFile,
    const std::string& folderName,
    const std::vector<int>& thresholdsAndStep,
    const std::vector<int>& colorNuances,
    const int fraction,
    std::vector<int>& rectanglesToModify,
    const std::vector<int>& tolerance,
    const bool severalColors,
    const bool totalBlackAndWhite,
    const bool totalReversal,
    const bool partial,
    const bool partialInDiagonal,
    const bool alternatingBlackAndWhite,
    const bool oneColor
) {

    // Extract the base name without the extension
    const size_t dotPos = inputFile.find_last_of('.');
    const std::string baseName = inputFile.substr(0, dotPos);

    // Load the image from the input path
    const std::string inputPath = "Input/" + folderName + "/" + inputFile;
    printf("%s\n", folderName.c_str()); // Print the folder name

    const Image image(inputPath.c_str());

    const int first_threshold = 3 * thresholdsAndStep[0];
    const int last_threshold = 3 * thresholdsAndStep[1];
    const int step = 3 * thresholdsAndStep[2];

    const int firstColorNuance = colorNuances[0];
    const int lastColorNuance = colorNuances[1];
    const int stepColorNuance = colorNuances[2];

    // One Color transformation
    if (oneColor) {
        oneColorTransformations(image, baseName, tolerance);
    }

    if (severalColors) {
        several_colors_transformations(image, baseName, first_threshold, last_threshold, step,
            firstColorNuance, lastColorNuance, stepColorNuance);
    }

    // Total Black and White
    if (totalBlackAndWhite) {
        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_black_and_white_transformations),
            total_black_and_white_output_dirs,
            total_black_and_white_suffixes,
            baseName,
            " - ",
            first_threshold,
            last_threshold,
            step,
            true,
            FOLDER_120,
            {}
        );
    }

    // Total Reversal
    if (totalReversal) {
        applyAndSaveGenericTransformations(
            image,
            wrapSimpleTransforms(total_reversal_step_by_step_transformations),
            reversal_step_by_step_output_dirs,
            reversal_step_by_step_suffixes,
            baseName,
            " - ",
            first_threshold,
            last_threshold,
            step,
            true,
            FOLDER_120,
            {}
        );
    }

    // Alternating Black and White
    if (alternatingBlackAndWhite) {
        std::vector<int> altParams = {first_threshold, last_threshold, step};
        applyAndSaveGenericTransformations(
            image,
            wrapAlternatingTransforms(total_alternating_black_and_white_transformations),
            total_step_by_step_output_dirs,
            total_step_by_step_suffixes,
            baseName,
            " Alternating - ",
            first_threshold,
            last_threshold,
            step,
            true,
            FOLDER_120,
            altParams
        );
    }

    // Partial Transformations
    if (partial) {
        several_colors_partial_transformations(
            image,
            wrapPartialTransformsWithRectangles(partialTransformationsFunc),
            baseName,
            first_threshold,
            last_threshold,
            step,
            fraction,
            rectanglesToModify,
            firstColorNuance,
            lastColorNuance,
            stepColorNuance
        );
    }

    if (partialInDiagonal) {
        std::vector<int> rectanglesInDiagonal = genererRectanglesInDiagonal(fraction);
        several_colors_partial_transformations(
            image,
            wrapPartialTransformsWithRectangles(partialTransformationsFunc),
            baseName,
            first_threshold,
            last_threshold,
            step,
            fraction,
            rectanglesInDiagonal,
            firstColorNuance,
            lastColorNuance,
            stepColorNuance
        );
    }

	edge_detector(image, baseName);

}

