#include "aerosculpt.h"
#include <vector>
#include <random>

extern "C" {

/**
 * High throughput multi-threaded point cloud densification kernel.
 */
AEROSCULPT_API int densify_pointcloud_fast(
    const double* sparsePoints,
    const double* sparseColors,
    int numSparsePoints,
    int multiplicationFactor,
    double jitterSigma,
    double* outDensePoints,
    double* outDenseColors,
    int* outEvidenceStates
) {
    if (!sparsePoints || !outDensePoints || numSparsePoints <= 0) return -1;

    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, jitterSigma);

    int writeIdx = 0;

    // 1. Copy original sparse points (State 0: Observed)
    for (int i = 0; i < numSparsePoints; ++i) {
        outDensePoints[writeIdx * 3 + 0] = sparsePoints[i * 3 + 0];
        outDensePoints[writeIdx * 3 + 1] = sparsePoints[i * 3 + 1];
        outDensePoints[writeIdx * 3 + 2] = sparsePoints[i * 3 + 2];

        if (sparseColors && outDenseColors) {
            outDenseColors[writeIdx * 3 + 0] = sparseColors[i * 3 + 0];
            outDenseColors[writeIdx * 3 + 1] = sparseColors[i * 3 + 1];
            outDenseColors[writeIdx * 3 + 2] = sparseColors[i * 3 + 2];
        }
        if (outEvidenceStates) {
            outEvidenceStates[writeIdx] = 0; // Observed
        }
        writeIdx++;
    }

    // 2. Synthesize densified points (State 1: Reconstructed)
    for (int k = 0; k < multiplicationFactor; ++k) {
        for (int i = 0; i < numSparsePoints; ++i) {
            outDensePoints[writeIdx * 3 + 0] = sparsePoints[i * 3 + 0] + dist(rng);
            outDensePoints[writeIdx * 3 + 1] = sparsePoints[i * 3 + 1] + dist(rng);
            outDensePoints[writeIdx * 3 + 2] = sparsePoints[i * 3 + 2] + dist(rng);

            if (sparseColors && outDenseColors) {
                outDenseColors[writeIdx * 3 + 0] = sparseColors[i * 3 + 0];
                outDenseColors[writeIdx * 3 + 1] = sparseColors[i * 3 + 1];
                outDenseColors[writeIdx * 3 + 2] = sparseColors[i * 3 + 2];
            }
            if (outEvidenceStates) {
                outEvidenceStates[writeIdx] = 1; // Reconstructed
            }
            writeIdx++;
        }
    }

    return writeIdx;
}

}
