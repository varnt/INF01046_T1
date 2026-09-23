
// INF01046 - Fundamentos de Processamento de Imagens
// Operacoes do Trabalho 1, reaproveitadas como base do Trabalho 2:
//   a) Espelhamento horizontal e vertical
//   b) Conversao para tons de cinza (luminancia)
//   c) Quantizacao de tons


#ifndef IMAGE_OPS_H
#define IMAGE_OPS_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
namespace imgops {

// Espelha a imagem horizontalmente e/ou verticalmente (memcpy por linha /
// por pixel). Funciona para numero par ou impar de linhas/colunas.
Mat mirror(const Mat& src, bool horizontal, bool vertical);

// Converte uma imagem colorida (BGR, 3 canais) para tons de cinza usando
// L = 0.299R + 0.587G + 0.114B. Retorna uma imagem de 3 canais com R=G=B=L.
Mat toGrayscaleLuminance(const Mat& src);

// Mesma conversao, mas retorna uma imagem de 1 canal (CV_8UC1), mais
// pratica para os calculos de histograma do Trabalho 2. Aceita entrada de
// 1 ou 3 canais (se ja for 1 canal, apenas clona).
Mat toGrayscaleLuminance1C(const Mat& src);

// Quantiza os tons de uma imagem em tons de cinza (3 canais, R=G=B) para,
// no maximo, "n" niveis distintos, pelo esquema de binning do enunciado.
Mat quantize(const Mat& grayImg, int n);

} // namespace imgops

#endif // IMAGE_OPS_H
