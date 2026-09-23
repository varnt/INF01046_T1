#include "PointOps.h"
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;
namespace pointops {

static inline uchar clampToByte(double v) {
    if (v < 0.0)   return 0;
    if (v > 255.0) return 255;
    return static_cast<uchar>(lround(v));
}

Mat adjustBrightness(const Mat& src, int delta) {
    delta = max(-255, min(255, delta));
    Mat dst(src.rows, src.cols, src.type());

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
            const Vec3b* s = src.ptr<Vec3b>(i);
            Vec3b* d = dst.ptr<Vec3b>(i);
            for (int j = 0; j < src.cols; ++j) {
                for (int c = 0; c < 3; ++c) {
                    d[j][c] = clampToByte(static_cast<double>(s[j][c]) + delta);
                }
            }
        }
    }
    return dst;
}

Mat adjustContrast(const Mat& src, double factor) {
    if (factor <= 0.0)   factor = 0.0001; // mantem o intervalo (0, 255]
    if (factor > 255.0)  factor = 255.0;
    Mat dst(src.rows, src.cols, src.type());

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
            const Vec3b* s = src.ptr<Vec3b>(i);
            Vec3b* d = dst.ptr<Vec3b>(i);
            for (int j = 0; j < src.cols; ++j) {
                for (int c = 0; c < 3; ++c) {
                    d[j][c] = clampToByte(static_cast<double>(s[j][c]) * factor);
                }
            }
        }
    }
    return dst;
}

Mat negative(const Mat& src) {
    Mat dst(src.rows, src.cols, src.type());

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
            const Vec3b* s = src.ptr<Vec3b>(i);
            Vec3b* d = dst.ptr<Vec3b>(i);
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
