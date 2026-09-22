#include "Histogram.h"
#include "ImageOps.h"
#include <algorithm>
#include <cmath>

namespace histo {

Hist256 computeHistogram(const cv::Mat& gray1C) {
    CV_Assert(gray1C.channels() == 1);
    Hist256 hist{};
    hist.fill(0);
    for (int i = 0; i < gray1C.rows; ++i) {
        const uchar* row = gray1C.ptr<uchar>(i);
        for (int j = 0; j < gray1C.cols; ++j) {
            hist[row[j]]++;
        }
    }
    return hist;
}

cv::Mat drawHistogram(const Hist256& hist, int width, int height) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    long maxVal = *std::max_element(hist.begin(), hist.end());
    if (maxVal <= 0) maxVal = 1;

    const int margin = 10;                 // margem superior
    const int usableHeight = height - margin;
    const double binWidth = static_cast<double>(width) / 256.0;

    for (int bin = 0; bin < 256; ++bin) {
        int x0 = static_cast<int>(std::round(bin * binWidth));
        int x1 = static_cast<int>(std::round((bin + 1) * binWidth));
        if (x1 <= x0) x1 = x0 + 1;
        x1 = std::min(x1, width);

        double normalized = static_cast<double>(hist[bin]) / static_cast<double>(maxVal);
        int barHeight = static_cast<int>(std::round(normalized * usableHeight));

        int yTop = height - 1 - barHeight;
        yTop = std::max(0, yTop);

        for (int y = yTop; y < height; ++y) {
            cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
            for (int x = x0; x < x1; ++x) {
                row[x] = cv::Vec3b(0, 0, 0);
            }
        }
    }
    return img;
}

std::array<uchar, 256> buildEqualizationMap(const Hist256& hist) {
    std::array<long, 256> cdf{};
    long running = 0;
    for (int v = 0; v < 256; ++v) {
        running += hist[v];
        cdf[v] = running;
    }
    long total = cdf[255];

    long cdfMin = 0;
    for (int v = 0; v < 256; ++v) {
        if (cdf[v] > 0) { cdfMin = cdf[v]; break; }
    }

    std::array<uchar, 256> map{};
    if (total <= cdfMin) {
        // imagem com um unico tom (ou vazia): mapeamento identidade
        for (int v = 0; v < 256; ++v) map[v] = static_cast<uchar>(v);
        return map;
    }

    for (int v = 0; v < 256; ++v) {
        double val = static_cast<double>(cdf[v] - cdfMin) /
                     static_cast<double>(total - cdfMin) * 255.0;
        int vi = static_cast<int>(std::lround(val));
        vi = std::max(0, std::min(255, vi));
        map[v] = static_cast<uchar>(vi);
    }
    return map;
}

cv::Mat equalizeGrayscale(const cv::Mat& gray1C, Hist256* histBefore) {
    Hist256 hist = computeHistogram(gray1C);
    if (histBefore) *histBefore = hist;

    std::array<uchar, 256> map = buildEqualizationMap(hist);

    cv::Mat dst(gray1C.rows, gray1C.cols, CV_8UC1);
    for (int i = 0; i < gray1C.rows; ++i) {
        const uchar* s = gray1C.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < gray1C.cols; ++j) {
            d[j] = map[s[j]];
        }
    }
    return dst;
}

cv::Mat equalizeColorViaLuminance(const cv::Mat& colorBGR) {
    CV_Assert(colorBGR.channels() == 3);

    cv::Mat luminance = imgops::toGrayscaleLuminance1C(colorBGR);
    Hist256 hist = computeHistogram(luminance);
    std::array<uchar, 256> map = buildEqualizationMap(hist);

    cv::Mat dst(colorBGR.rows, colorBGR.cols, colorBGR.type());
    for (int i = 0; i < colorBGR.rows; ++i) {
        const cv::Vec3b* s = colorBGR.ptr<cv::Vec3b>(i);
        cv::Vec3b* d = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < colorBGR.cols; ++j) {
            d[j][0] = map[s[j][0]]; // B
            d[j][1] = map[s[j][1]]; // G
            d[j][2] = map[s[j][2]]; // R
        }
    }
    return dst;
}

cv::Mat histogramMatching(const cv::Mat& src1C, const cv::Mat& reference1C) {
    Hist256 hs = computeHistogram(src1C);
    Hist256 hr = computeHistogram(reference1C);

    double totalS = 0.0, totalR = 0.0;
    for (int v = 0; v < 256; ++v) { totalS += hs[v]; totalR += hr[v]; }
    if (totalS <= 0) totalS = 1;
    if (totalR <= 0) totalR = 1;

    std::array<double, 256> cdfS{}, cdfR{};
    double runS = 0.0, runR = 0.0;
    for (int v = 0; v < 256; ++v) {
        runS += hs[v]; cdfS[v] = runS / totalS;
        runR += hr[v]; cdfR[v] = runR / totalR;
    }

    // Para cada nivel de cinza v da origem, procura o nivel g da referencia
    // cujo histograma cumulativo mais se aproxima do de v.
    std::array<uchar, 256> map{};
    for (int v = 0; v < 256; ++v) {
        double best = 1e18;
        int bestG = 0;
        for (int g = 0; g < 256; ++g) {
            double diff = std::fabs(cdfS[v] - cdfR[g]);
            if (diff < best) { best = diff; bestG = g; }
        }
        map[v] = static_cast<uchar>(bestG);
    }

    cv::Mat dst(src1C.rows, src1C.cols, CV_8UC1);
    for (int i = 0; i < src1C.rows; ++i) {
        const uchar* s = src1C.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src1C.cols; ++j) {
            d[j] = map[s[j]];
        }
    }
    return dst;
}

} // namespace histo
