// ============================================================================
// INF01046 - Trabalho 2 - Parte 1, itens 2, 3 e 4
// Operacoes de ponto (pixel a pixel), aplicadas a cada canal (R,G,B) de
// forma independente no caso de imagens coloridas.
// ============================================================================

#ifndef POINT_OPS_H
#define POINT_OPS_H

#include <opencv2/opencv.hpp>

namespace pointops {

// (2) Ajuste de brilho: novoValor = clamp(valor + delta, 0, 255)
// delta deve estar no intervalo [-255, 255].
cv::Mat adjustBrightness(const cv::Mat& src, int delta);

// (3) Ajuste de contraste: novoValor = clamp(valor * factor, 0, 255)
// factor deve estar no intervalo (0, 255].
cv::Mat adjustContrast(const cv::Mat& src, double factor);

// (4) Negativo: novoValor = 255 - valor
cv::Mat negative(const cv::Mat& src);

} // namespace pointops

#endif // POINT_OPS_H
