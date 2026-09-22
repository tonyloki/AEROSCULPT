#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>

#ifdef _WIN32
    #ifdef AEROSCULPT_EXPORTS
        #define AEROSCULPT_API __declspec(dllexport)
    #else
        #define AEROSCULPT_API __declspec(dllimport)
    #endif
#else
    #define AEROSCULPT_API __attribute__((visibility("default")))
#endif

extern "C" {

/**
 * High-speed multi-threaded blur assessment using OpenMP and OpenCV.
 * Calculates Laplacian variance for high-framerate drone video feeds.
 */
AEROSCULPT_API double calculate_laplacian_variance_fast(const unsigned char* imageData, int width, int height);

/**
 * Rapid exposure and histogram balance assessment.
 */
AEROSCULPT_API bool evaluate_frame_exposure_fast(const unsigned char* imageData, int width, int height, double* outMeanLuminance);

/**
 * 7-DOF Sim(3) Umeyama closed-form metric alignment solver.
 */
AEROSCULPT_API int solve_sim3_alignment(
    const double* srcPoints,
    const double* tgtPoints,
    int numPoints,
    double* outScale,
    double* outRotation3x3,
    double* outTranslation3,
    double* outRmse
);

}
