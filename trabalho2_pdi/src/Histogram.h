// ============================================================================
// INF01046 - Trabalho 2 - Parte 1, itens 1, 5 e 6
// Histograma, equalizacao e histogram matching.
// ============================================================================

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/opencv.hpp>
#include <array>

namespace histo {

using Hist256 = std::array<long, 256>;

// (1) Calcula o histograma de uma imagem em tons de cinza (1 canal, 8 bits).
Hist256 computeHistogram(const cv::Mat& gray1C);

// (1) Desenha o histograma como uma imagem (fundo branco, barras pretas),
// normalizando a altura das colunas pelo valor maximo do histograma.
cv::Mat drawHistogram(const Hist256& hist, int width = 512, int height = 400);

// Constroi a tabela de mapeamento de equalizacao (256 valores) a partir de
// um histograma, usando o histograma cumulativo classico:
//   mapa[v] = round( (cdf[v] - cdf_min) / (N - cdf_min) * 255 )
std::array<uchar, 256> buildEqualizationMap(const Hist256& hist);

// (5) Equaliza uma imagem em tons de cinza (1 canal). Retorna a imagem
// equalizada; se histOut != nullptr, tambem devolve o histograma da
// imagem ANTES da equalizacao (util para exibir antes/depois).
cv::Mat equalizeGrayscale(const cv::Mat& gray1C, Hist256* histBefore = nullptr);

// (5) Equaliza uma imagem colorida: calcula o histograma cumulativo a
// partir da imagem de luminancia e aplica O MESMO mapeamento a cada canal
// (R, G, B) independentemente, conforme pedido no enunciado.
cv::Mat equalizeColorViaLuminance(const cv::Mat& colorBGR);

// (6) Histogram matching entre duas imagens em tons de cinza (1 canal):
// transforma "src" para que seu histograma se aproxime do histograma de
// "reference".
cv::Mat histogramMatching(const cv::Mat& src1C, const cv::Mat& reference1C);

} // namespace histo

#endif // HISTOGRAM_H
