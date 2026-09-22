#include "Convolution.h"
#include <cmath>
#include <algorithm>

namespace conv {

std::vector<Kernel3x3> builtinKernels() {
    std::vector<Kernel3x3> ks;

    // (i) Gaussiano - passa baixas (borramento). Pode ser aplicado direto
    // em imagem colorida; nao soma offset (pesos somam 1, sempre >=0).
    ks.push_back(Kernel3x3{
        "i - Gaussiano (passa-baixas)",
        {{0.0625, 0.125, 0.0625},
         {0.125,  0.25,  0.125},
         {0.0625, 0.125, 0.0625}},
        false, true
    });

    // (ii) Laplaciano - passa altas (detecta arestas).
    ks.push_back(Kernel3x3{
        "ii - Laplaciano (passa-altas)",
        {{ 0, -1,  0},
         {-1,  4, -1},
         { 0, -1,  0}},
        false, false
    });

    // (iii) Passa altas generico (detector de arestas mais sensivel).
    ks.push_back(Kernel3x3{
        "iii - Passa-altas generico",
        {{-1, -1, -1},
         {-1,  8, -1},
         {-1, -1, -1}},
        false, false
    });

    // (iv) Prewitt Hx.
    ks.push_back(Kernel3x3{
        "iv - Prewitt Hx",
        {{-1, 0, 1},
         {-1, 0, 1},
         {-1, 0, 1}},
        true, false
    });

    // (v) Prewitt Hy.
    ks.push_back(Kernel3x3{
        "v - Prewitt Hy",
        {{-1, -1, -1},
         { 0,  0,  0},
         { 1,  1,  1}},
        true, false
    });

    // (vi) Sobel Hx.
    ks.push_back(Kernel3x3{
        "vi - Sobel Hx",
        {{-1, 0, 1},
         {-2, 0, 2},
         {-1, 0, 1}},
        true, false
    });

    // (vii) Sobel Hy.
    ks.push_back(Kernel3x3{
        "vii - Sobel Hy",
        {{-1, -2, -1},
         { 0,  0,  0},
         { 1,  2,  1}},
        true, false
    });

    return ks;
}

static inline uchar clampToByte(double v) {
    if (v < 0.0)   return 0;
    if (v > 255.0) return 255;
    return static_cast<uchar>(std::lround(v));
}

// Conv(E) = i*A + h*B + g*C + f*D + e*E + d*F + c*G + b*H + a*I, com o
// kernel k=[[a,b,c],[d,e,f],[g,h,i]]. Equivale a:
//   Conv(r,c) = soma, para di,dj em {-1,0,1}, de k[1-di][1-dj] * img(r+di,c+dj)
static inline double convAt3x3(const uchar* rowAbove, const uchar* rowMid,
                                const uchar* rowBelow, int c,
                                const double kernel[3][3]) {
    double sum = 0.0;
    sum += kernel[1 - (-1)][1 - (-1)] * rowAbove[c - 1]; // di=-1,dj=-1 -> k[2][2]
    sum += kernel[1 - (-1)][1 - ( 0)] * rowAbove[c];     // di=-1,dj= 0 -> k[2][1]
    sum += kernel[1 - (-1)][1 - ( 1)] * rowAbove[c + 1]; // di=-1,dj= 1 -> k[2][0]
    sum += kernel[1 - ( 0)][1 - (-1)] * rowMid[c - 1];   // di= 0,dj=-1 -> k[1][2]
    sum += kernel[1 - ( 0)][1 - ( 0)] * rowMid[c];       // di= 0,dj= 0 -> k[1][1]
    sum += kernel[1 - ( 0)][1 - ( 1)] * rowMid[c + 1];   // di= 0,dj= 1 -> k[1][0]
    sum += kernel[1 - ( 1)][1 - (-1)] * rowBelow[c - 1]; // di= 1,dj=-1 -> k[0][2]
    sum += kernel[1 - ( 1)][1 - ( 0)] * rowBelow[c];     // di= 1,dj= 0 -> k[0][1]
    sum += kernel[1 - ( 1)][1 - ( 1)] * rowBelow[c + 1]; // di= 1,dj= 1 -> k[0][0]
    return sum;
}

cv::Mat convolve3x3Gray(const cv::Mat& gray1C, const double kernel[3][3], bool addOffset127) {
    CV_Assert(gray1C.channels() == 1);
    cv::Mat dst = gray1C.clone(); // bordas permanecem iguais a original

    const int rows = gray1C.rows;
    const int cols = gray1C.cols;

    for (int r = 1; r < rows - 1; ++r) {
        const uchar* rowAbove = gray1C.ptr<uchar>(r - 1);
        const uchar* rowMid   = gray1C.ptr<uchar>(r);
        const uchar* rowBelow = gray1C.ptr<uchar>(r + 1);
        uchar* dstRow = dst.ptr<uchar>(r);

        for (int c = 1; c < cols - 1; ++c) {
            double sum = convAt3x3(rowAbove, rowMid, rowBelow, c, kernel);
            if (addOffset127) sum += 127.0;
            dstRow[c] = clampToByte(sum);
        }
    }
    return dst;
}

cv::Mat convolve3x3Color(const cv::Mat& colorBGR, const double kernel[3][3], bool addOffset127) {
    CV_Assert(colorBGR.channels() == 3);
    cv::Mat dst = colorBGR.clone();

    const int rows = colorBGR.rows;
    const int cols = colorBGR.cols;

    for (int r = 1; r < rows - 1; ++r) {
        const cv::Vec3b* rowAbove = colorBGR.ptr<cv::Vec3b>(r - 1);
        const cv::Vec3b* rowMid   = colorBGR.ptr<cv::Vec3b>(r);
        const cv::Vec3b* rowBelow = colorBGR.ptr<cv::Vec3b>(r + 1);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(r);

        for (int c = 1; c < cols - 1; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                double sum = 0.0;
                sum += kernel[2][2] * rowAbove[c - 1][ch];
                sum += kernel[2][1] * rowAbove[c][ch];
                sum += kernel[2][0] * rowAbove[c + 1][ch];
                sum += kernel[1][2] * rowMid[c - 1][ch];
                sum += kernel[1][1] * rowMid[c][ch];
                sum += kernel[1][0] * rowMid[c + 1][ch];
                sum += kernel[0][2] * rowBelow[c - 1][ch];
                sum += kernel[0][1] * rowBelow[c][ch];
                sum += kernel[0][0] * rowBelow[c + 1][ch];
                if (addOffset127) sum += 127.0;
                dstRow[c][ch] = clampToByte(sum);
            }
        }
    }
    return dst;
}

} // namespace conv
