
// INF01046 - Trabalho 2 - Parte 1, item 5 (PONTOS EXTRA - 20 pontos)
// Equalizacao de histograma no espaco de cor L*a*b*.
//
// Conversao RGB <-> L*a*b* implementada manualmente (sem cvtColor),
// via CIE XYZ (D65), seguindo as formulas padrao do espaco de cor.


#ifndef LAB_OPS_H
#define LAB_OPS_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
namespace labops {

// Converte um pixel BGR (0-255 cada) para Lab (L em [0,100], a,b tipicamente
// em [-128,127]).
void bgrToLab(uchar B, uchar G, uchar R, double& L, double& a, double& b);

// Converte um pixel Lab de volta para BGR (0-255, com clamping).
void labToBgr(double L, double a, double b, uchar& B, uchar& G, uchar& R);

// Equaliza uma imagem colorida no espaco Lab: converte para Lab, equaliza
// o histograma do canal L (via histograma cumulativo classico), mantem a
// e b inalterados, e converte de volta para BGR.
Mat equalizeLab(const Mat& colorBGR);

} // namespace labops

#endif // LAB_OPS_H
