#include "PointOps.h"
#include <algorithm>
#include <cmath>

namespace pointops {

static inline uchar clampToByte(double v) {
    if (v < 0.0)   return 0;
    if (v > 255.0) return 255;
    return static_cast<uchar>(std::lround(v));
}

cv::Mat adjustBrightness(const cv::Mat& src, int delta) {
    delta = std::max(-255, std::min(255, delta));
    cv::Mat dst(src.rows, src.cols, src.type());

    if (src.channels() == 1) {
        for (int i = 0; i < src.rows; ++i) {
            const uchar* s = src.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            for (int j = 0; j < src.cols; ++j) {
                d[j] = clampToByte(static_cast<double>(s[j]) + delta);
            }
        }
    } else {
        CV_Assert(src.channels() == 3);
        for (int i = 0; i < src.rows; ++i) {
            const cv::Vec3b* s = src.ptr<cv::Vec3b>(i);
            cv::Vec3b* d = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < src.cols; ++j) {
                for (int c = 0; c < 3; ++c) {
                    d[j][c] = clampToByte(static_cast<double>(s[j][c]) + delta);
                }
            }
        }
    }
    return dst;
}

cv::Mat adjustContrast(const cv::Mat& src, double factor) {
    if (factor <= 0.0)   factor = 0.0001; // mantem o intervalo (0, 255]
    if (factor > 255.0)  factor = 255.0;
    cv::Mat dst(src.rows, src.cols, src.type());

    if (src.channels() == 1) {
        for (int i = 0; i < src.rows; ++i) {
            const uchar* s = src.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            for (int j = 0; j < src.cols; ++j) {
                d[j] = clampToByte(static_cast<double>(s[j]) * factor);
            }
        }
    } else {
        CV_Assert(src.channels() == 3);
        for (int i = 0; i < src.rows; ++i) {
            const cv::Vec3b* s = src.ptr<cv::Vec3b>(i);
            cv::Vec3b* d = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < src.cols; ++j) {
                for (int c = 0; c < 3; ++c) {
                    d[j][c] = clampToByte(static_cast<double>(s[j][c]) * factor);
                }
            }
        }
    }
    return dst;
}

cv::Mat negative(const cv::Mat& src) {
    cv::Mat dst(src.rows, src.cols, src.type());

    if (src.channels() == 1) {
        for (int i = 0; i < src.rows; ++i) {
            const uchar* s = src.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            for (int j = 0; j < src.cols; ++j) {
                d[j] = static_cast<uchar>(255 - s[j]);
            }
        }
    } else {
        CV_Assert(src.channels() == 3);
        for (int i = 0; i < src.rows; ++i) {
            const cv::Vec3b* s = src.ptr<cv::Vec3b>(i);
            cv::Vec3b* d = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < src.cols; ++j) {
                for (int c = 0; c < 3; ++c) {
                    d[j][c] = static_cast<uchar>(255 - s[j][c]);
                }
            }
        }
    }
    return dst;
}

} // namespace pointops
