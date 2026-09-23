#include "ImageOps.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;
namespace imgops {

Mat mirror(const Mat& src, bool horizontal, bool vertical) {
    Mat dst = src.clone();
    const int rows = dst.rows;
    const int cols = dst.cols;
    const size_t elemSize = dst.elemSize();

    if (vertical) {
        vector<uchar> temp(dst.step);
        for (int i = 0; i < rows / 2; ++i) {
            uchar* rowTop = dst.ptr<uchar>(i);
            uchar* rowBot = dst.ptr<uchar>(rows - 1 - i);
            memcpy(temp.data(), rowTop, dst.step);
            memcpy(rowTop, rowBot, dst.step);
            memcpy(rowBot, temp.data(), dst.step);
        }
    }

    if (horizontal) {
        vector<uchar> temp(elemSize);
        for (int i = 0; i < rows; ++i) {
            uchar* rowPtr = dst.ptr<uchar>(i);
            for (int j = 0; j < cols / 2; ++j) {
                uchar* pixelLeft  = rowPtr + j * elemSize;
                uchar* pixelRight = rowPtr + (cols - 1 - j) * elemSize;
                memcpy(temp.data(), pixelLeft, elemSize);
                memcpy(pixelLeft, pixelRight, elemSize);
                memcpy(pixelRight, temp.data(), elemSize);
            }
        }
    }

    return dst;
}

Mat toGrayscaleLuminance(const Mat& src) {
    CV_Assert(src.channels() == 3);
    Mat dst(src.rows, src.cols, src.type());

    for (int i = 0; i < src.rows; ++i) {
        const Vec3b* srcRow = src.ptr<Vec3b>(i);
        Vec3b* dstRow = dst.ptr<Vec3b>(i);
        for (int j = 0; j < src.cols; ++j) {
            double B = srcRow[j][0];
            double G = srcRow[j][1];
            double R = srcRow[j][2];
            double L = 0.299 * R + 0.587 * G + 0.114 * B;
            int Li = static_cast<int>(lround(L));
            Li = min(255, max(0, Li));
            uchar Lc = static_cast<uchar>(Li);
            dstRow[j] = Vec3b(Lc, Lc, Lc);
        }
    }
    return dst;
}

Mat toGrayscaleLuminance1C(const Mat& src) {
    if (src.channels() == 1) {
        return src.clone();
    }
    CV_Assert(src.channels() == 3);
    Mat dst(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; ++i) {
        const Vec3b* srcRow = src.ptr<Vec3b>(i);
        uchar* dstRow = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j) {
            double B = srcRow[j][0];
            double G = srcRow[j][1];
            double R = srcRow[j][2];
            double L = 0.299 * R + 0.587 * G + 0.114 * B;
            int Li = static_cast<int>(lround(L));
            Li = min(255, max(0, Li));
            dstRow[j] = static_cast<uchar>(Li);
        }
    }
    return dst;
}

Mat quantize(const Mat& grayImg, int n) {
    CV_Assert(grayImg.channels() == 3);
    if (n <= 0) return grayImg.clone();

    int t1 = 255, t2 = 0;
    for (int i = 0; i < grayImg.rows; ++i) {
        const Vec3b* row = grayImg.ptr<Vec3b>(i);
        for (int j = 0; j < grayImg.cols; ++j) {
            int v = row[j][0];
            if (v < t1) t1 = v;
            if (v > t2) t2 = v;
        }
    }

    int tam_int = t2 - t1 + 1;
    if (n >= tam_int) return grayImg.clone();

    double tb = static_cast<double>(tam_int) / static_cast<double>(n);
    Mat dst(grayImg.rows, grayImg.cols, grayImg.type());

    for (int i = 0; i < grayImg.rows; ++i) {
        const Vec3b* srcRow = grayImg.ptr<Vec3b>(i);
        Vec3b* dstRow = dst.ptr<Vec3b>(i);
        for (int j = 0; j < grayImg.cols; ++j) {
            double t_orig = srcRow[j][0];
            int binIdx = static_cast<int>(floor((t_orig - (t1 - 0.5)) / tb));
            if (binIdx < 0)  binIdx = 0;
            if (binIdx >= n) binIdx = n - 1;
            double binStart = (t1 - 0.5) + binIdx * tb;
            double center   = binStart + tb / 2.0;
            int q = static_cast<int>(lround(center));
            q = min(255, max(0, q));
            uchar qc = static_cast<uchar>(q);
            dstRow[j] = Vec3b(qc, qc, qc);
        }
    }
    return dst;
}

} // namespace imgops
