// ============================================================================
// INF01046 - Trabalho 2 - Parte 2, itens 7, 8 e 9
// Reducao (zoom out), ampliacao 2x2 (zoom in) e rotacao de 90 graus.
// ============================================================================

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <opencv2/opencv.hpp>

namespace geom {

// (7) Reduz a imagem usando fatores de reducao sx (colunas) e sy (linhas),
// ambos >= 1, nao necessariamente iguais. Cada pixel de saida e a media
// dos pixels da imagem original cobertos pelo retangulo sy x sx
// correspondente (recortado nas bordas quando necessario).
cv::Mat zoomOut(const cv::Mat& src, double sx, double sy);

// (8) Amplia a imagem por um fator fixo de 2 em cada dimensao (a imagem de
// saida tem 4x o numero de pixels da original). Implementado em dois
// passos 1-D: insere linhas/colunas em branco e depois interpola
// linearmente ao longo das linhas e, em seguida, das colunas.
cv::Mat zoomIn2x(const cv::Mat& src);

// (9) Rotaciona a imagem 90 graus. clockwise = true -> sentido horario;
// false -> sentido anti-horario. Chamar a funcao multiplas vezes permite
// obter rotacoes de +/-180, +/-270, etc.
cv::Mat rotate90(const cv::Mat& src, bool clockwise);

} // namespace geom

#endif // GEOMETRY_H
