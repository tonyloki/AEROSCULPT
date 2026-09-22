#include "aerosculpt.h"
#include <opencv2/core.hpp>
#include <cmath>
#include <vector>

extern "C" {

AEROSCULPT_API int solve_sim3_alignment(
    const double* srcPoints,
    const double* tgtPoints,
    int numPoints,
    double* outScale,
    double* outRotation3x3,
    double* outTranslation3,
    double* outRmse
) {
    if (numPoints < 3 || !srcPoints || !tgtPoints) return -1;

    cv::Mat X(numPoints, 3, CV_64F, const_cast<double*>(srcPoints));
    cv::Mat Y(numPoints, 3, CV_64F, const_cast<double*>(tgtPoints));

    cv::Scalar meanX_s = cv::mean(X);
    cv::Scalar meanY_s = cv::mean(Y);

    cv::Mat muX = (cv::Mat_<double>(1, 3) << meanX_s[0], meanX_s[1], meanX_s[2]);
    cv::Mat muY = (cv::Mat_<double>(1, 3) << meanY_s[0], meanY_s[1], meanY_s[2]);

    cv::Mat X_c = X - cv::repeat(muX, numPoints, 1);
    cv::Mat Y_c = Y - cv::repeat(muY, numPoints, 1);

    double varX = 0.0;
    for (int i = 0; i < numPoints; ++i) {
        double x0 = X_c.at<double>(i, 0);
        double x1 = X_c.at<double>(i, 1);
        double x2 = X_c.at<double>(i, 2);
        varX += (x0 * x0 + x1 * x1 + x2 * x2);
    }
    varX /= numPoints;

    cv::Mat Sigma = (Y_c.t() * X_c) / (double)numPoints;

    cv::Mat w, u, vt;
    cv::SVDecomp(Sigma, w, u, vt);

    cv::Mat S = cv::Mat::eye(3, 3, CV_64F);
    double detU = cv::determinant(u);
    double detV = cv::determinant(vt.t());
    if (detU * detV < 0.0) {
        S.at<double>(2, 2) = -1.0;
    }

    cv::Mat R = u * S * vt;
    double trDS = 0.0;
    for (int i = 0; i < 3; ++i) {
        trDS += w.at<double>(i, 0) * S.at<double>(i, i);
    }

    double scale = (varX > 1e-9) ? (trDS / varX) : 1.0;
    cv::Mat t = muY.t() - scale * (R * muX.t());

    if (outScale) *outScale = scale;
    if (outRotation3x3) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                outRotation3x3[r * 3 + c] = R.at<double>(r, c);
            }
        }
    }
    if (outTranslation3) {
        for (int i = 0; i < 3; ++i) {
            outTranslation3[i] = t.at<double>(i, 0);
        }
    }

    // RMSE computation
    double sumSq = 0.0;
    for (int i = 0; i < numPoints; ++i) {
        cv::Mat p = (cv::Mat_<double>(3, 1) << X.at<double>(i, 0), X.at<double>(i, 1), X.at<double>(i, 2));
        cv::Mat p_trans = scale * (R * p) + t;
        double dx = Y.at<double>(i, 0) - p_trans.at<double>(0, 0);
        double dy = Y.at<double>(i, 1) - p_trans.at<double>(1, 0);
        double dz = Y.at<double>(i, 2) - p_trans.at<double>(2, 0);
        sumSq += (dx * dx + dy * dy + dz * dz);
    }
    if (outRmse) {
        *outRmse = std::sqrt(sumSq / numPoints);
    }

    return 0;
}

}
