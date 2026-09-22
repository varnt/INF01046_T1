// ============================================================================
// INF01046 - Trabalho 2 - Parte 2, item 10
// Convolucao com filtro 3x3 arbitrario.
//
// Conv(E) = i*A + h*B + g*C + f*D + e*E + d*F + c*G + b*H + a*I
// (kernel k = [[a,b,c],[d,e,f],[g,h,i]] rotacionado 180 graus em relacao
// a vizinhanca 3x3 [[A,B,C],[D,E,F],[G,H,I]] em torno do pixel E)
//
// Bordas da imagem sao ignoradas (mantidas iguais a original); a
// convolucao e aplicada apenas ao interior da imagem.
// ============================================================================

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <vector>

namespace conv {

struct Kernel3x3 {
    std::string name;
    double w[3][3];   // w[i][j], i = linha (0=topo), j = coluna (0=esquerda)
    bool addOffset127;    // soma 127 antes do clamping (filtros iv-vii)
    bool colorAllowed;    // pode ser aplicado direto em imagem colorida (kernel i)
};

// Kernels pedidos no enunciado, na ordem (i) a (vii).
std::vector<Kernel3x3> builtinKernels();

// Aplica a convolucao 3x3 a uma imagem de 1 canal (luminancia), ignorando
// a borda de 1 pixel. addOffset127 controla se soma-se 127 ao resultado
// antes do clamping para [0,255] (usado nos filtros de gradiente).
cv::Mat convolve3x3Gray(const cv::Mat& gray1C, const double kernel[3][3], bool addOffset127);

// Aplica a convolucao 3x3 a uma imagem colorida (BGR), canal a canal,
// ignorando a borda de 1 pixel. Usado apenas para o filtro passa-baixas
// (Gaussiano), que pode ser aplicado diretamente sobre a imagem colorida.
cv::Mat convolve3x3Color(const cv::Mat& colorBGR, const double kernel[3][3], bool addOffset127);

} // namespace conv

#endif // CONVOLUTION_H
