#include "aerosculpt.h"
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <cmath>

extern "C" {

AEROSCULPT_API double calculate_laplacian_variance_fast(const unsigned char* imageData, int width, int height) {
    if (!imageData || width <= 0 || height <= 0) return 0.0;

    cv::Mat gray(height, width, CV_8UC1, const_cast<unsigned char*>(imageData));
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);

    double variance = stddev.val[0] * stddev.val[0];
    return variance;
}

AEROSCULPT_API bool evaluate_frame_exposure_fast(const unsigned char* imageData, int width, int height, double* outMeanLuminance) {
    if (!imageData || width <= 0 || height <= 0) return false;

    cv::Mat gray(height, width, CV_8UC1, const_cast<unsigned char*>(imageData));
    cv::Scalar meanVal = cv::mean(gray);

    if (outMeanLuminance) {
        *outMeanLuminance = meanVal.val[0];
    }

    // Underexposure threshold 35.0, Overexposure 225.0
    return (meanVal.val[0] > 35.0 && meanVal.val[0] < 225.0);
}

}
