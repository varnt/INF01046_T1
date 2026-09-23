#include "LabOps.h"
#include "Histogram.h"
#include <cmath>
#include <algorithm>
using namespace std;
using namespace cv;
namespace labops {

//constantes do espaco de cor (referencia de branco D65)
static const double Xn = 0.95047;
static const double Yn = 1.00000;
static const double Zn = 1.08883;

static inline double srgbToLinear(double c) {
    // c em [0,1]
    if (c <= 0.04045) return c / 12.92;
    return pow((c + 0.055) / 1.055, 2.4);
}

static inline double linearToSrgb(double c) {
    if (c <= 0.0031308) return c * 12.92;
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

static inline double fLab(double t) {
    const double delta = 6.0 / 29.0;
    if (t > delta * delta * delta) {
        return cbrt(t);
    }
    return t / (3.0 * delta * delta) + 4.0 / 29.0;
}

static inline double fLabInv(double t) {
    const double delta = 6.0 / 29.0;
    if (t > delta) {
        return t * t * t;
    }
    return 3.0 * delta * delta * (t - 4.0 / 29.0);
}

void bgrToLab(uchar Bc, uchar Gc, uchar Rc, double& L, double& a, double& b) {
    double Rn = Rc / 255.0, Gn = Gc / 255.0, Bn = Bc / 255.0;

    double Rl = srgbToLinear(Rn);
    double Gl = srgbToLinear(Gn);
    double Bl = srgbToLinear(Bn);

    // sRGB (linear) -> CIE XYZ (D65)
    double X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl;
    double Y = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl;
    double Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl;

    double fx = fLab(X / Xn);
    double fy = fLab(Y / Yn);
    double fz = fLab(Z / Zn);

    L = 116.0 * fy - 16.0;
    a = 500.0 * (fx - fy);
    b = 200.0 * (fy - fz);
}

void labToBgr(double L, double a, double b, uchar& Bc, uchar& Gc, uchar& Rc) {
    double fy = (L + 16.0) / 116.0;
    double fx = fy + a / 500.0;
    double fz = fy - b / 200.0;

    double X = Xn * fLabInv(fx);
    double Y = Yn * fLabInv(fy);
    double Z = Zn * fLabInv(fz);

    // CIE XYZ -> sRGB (linear)
    double Rl =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    double Gl = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
    double Bl =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;

    double Rn = linearToSrgb(Rl);
    double Gn = linearToSrgb(Gl);
    double Bn = linearToSrgb(Bl);

    auto toByte = [](double v) -> uchar {
        v = max(0.0, min(1.0, v));
        return static_cast<uchar>(lround(v * 255.0));
    };

    Rc = toByte(Rn);
    Gc = toByte(Gn);
    Bc = toByte(Bn);
}

Mat equalizeLab(const Mat& colorBGR) {
    CV_Assert(colorBGR.channels() == 3);
    const int rows = colorBGR.rows, cols = colorBGR.cols;

    // 1) Converte toda a imagem para Lab, guardando L (reescalado p/ 0-255
    //    para podermos usar o mesmo mecanismo de histograma/equalizacao
    //    ja implementado) e a, b em ponto flutuante.
    Mat L8(rows, cols, CV_8UC1);
    vector<double> aChannel(rows * cols), bChannel(rows * cols);

    for (int i = 0; i < rows; ++i) {
        const Vec3b* src = colorBGR.ptr<Vec3b>(i);
        uchar* Lrow = L8.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            double L, a, b;
            bgrToLab(src[j][0], src[j][1], src[j][2], L, a, b);
            int idx = i * cols + j;
            aChannel[idx] = a;
            bChannel[idx] = b;
            int L255 = static_cast<int>(lround(L / 100.0 * 255.0));
            L255 = max(0, min(255, L255));
            Lrow[j] = static_cast<uchar>(L255);
        }
    }

    // 2) Equaliza o canal L (reescalado 0-255) usando o mesmo mecanismo
    //    de histograma/equalizacao classico.
    histo::Hist256 hist = histo::computeHistogram(L8);
    array<uchar, 256> map = histo::buildEqualizationMap(hist);

    // 3) Reconstroi a imagem: L equalizado (convertido de volta para 0-100)
    //    + a,b originais -> BGR.
    Mat dst(rows, cols, colorBGR.type());
    for (int i = 0; i < rows; ++i) {
        const uchar* Lrow = L8.ptr<uchar>(i);
        Vec3b* d = dst.ptr<Vec3b>(i);
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            uchar Leq255 = map[Lrow[j]];
            double Leq = static_cast<double>(Leq255) / 255.0 * 100.0;

            uchar B, G, R;
            labToBgr(Leq, aChannel[idx], bChannel[idx], B, G, R);
            d[j] = Vec3b(B, G, R);
        }
    }
    return dst;
}

} // namespace labops
