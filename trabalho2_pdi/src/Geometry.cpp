#include "Geometry.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;
namespace geom {


// (7) Zoom out: reducao por medias de blocos sy x sx (nao necessariamente
// inteiros), sem sobreposicao e cobrindo toda a imagem. Funciona de forma
// genérica para imagens de 1 ou 3 canais operando byte a byte (cada byte
// corresponde a um canal de cor).

Mat zoomOut(const Mat& src, double sx, double sy) {
    CV_Assert(sx >= 1.0 && sy >= 1.0);

    const int rows = src.rows;
    const int cols = src.cols;
    const int elemSize = static_cast<int>(src.elemSize());

    const int outRows = static_cast<int>(ceil(rows / sy));
    const int outCols = static_cast<int>(ceil(cols / sx));

    Mat dst(outRows, outCols, src.type());

    for (int oi = 0; oi < outRows; ++oi) {
        int rowStart = static_cast<int>(round(oi * sy));
        int rowEnd   = static_cast<int>(round((oi + 1) * sy));
        rowStart = max(0, min(rows, rowStart));
        rowEnd   = max(rowStart + 1, min(rows, rowEnd));

        uchar* dstRow = dst.ptr<uchar>(oi);

        for (int oj = 0; oj < outCols; ++oj) {
            int colStart = static_cast<int>(round(oj * sx));
            int colEnd   = static_cast<int>(round((oj + 1) * sx));
            colStart = max(0, min(cols, colStart));
            colEnd   = max(colStart + 1, min(cols, colEnd));

            vector<double> sum(elemSize, 0.0);
            long count = 0;

            for (int r = rowStart; r < rowEnd; ++r) {
                const uchar* srcRow = src.ptr<uchar>(r);
                for (int c = colStart; c < colEnd; ++c) {
                    const uchar* px = srcRow + c * elemSize;
                    for (int e = 0; e < elemSize; ++e) sum[e] += px[e];
                    ++count;
                }
            }

            uchar* dstPx = dstRow + oj * elemSize;
            for (int e = 0; e < elemSize; ++e) {
                dstPx[e] = (count > 0)
                    ? static_cast<uchar>(lround(sum[e] / static_cast<double>(count)))
                    : 0;
            }
        }
    }
    return dst;
}


// (8) Zoom in 2x2, em dois passos 1-D:
//   1) posiciona os pixels originais nas posicoes pares (2i, 2j);
//   2) interpola horizontalmente ao longo das linhas pares (preenche as
//      colunas impares);
//   3) interpola verticalmente ao longo de TODAS as colunas (preenche as
//      linhas impares), usando os valores ja calculados no passo 2.
// A ultima linha/coluna (sem par seguinte para interpolar) repete o
// ultimo valor valido.

Mat zoomIn2x(const Mat& src) {
    const int R = src.rows;
    const int C = src.cols;
    const int elemSize = static_cast<int>(src.elemSize());

    Mat dst(2 * R, 2 * C, src.type(), Scalar::all(0));

    // Passo 1: posiciona os pixels originais em (2i, 2j).
    for (int i = 0; i < R; ++i) {
        const uchar* srcRow = src.ptr<uchar>(i);
        uchar* dstRow = dst.ptr<uchar>(2 * i);
        for (int j = 0; j < C; ++j) {
            memcpy(dstRow + (2 * j) * elemSize, srcRow + j * elemSize, elemSize);
        }
    }

    // Passo 2: interpolacao horizontal nas linhas pares (colunas impares).
    for (int i = 0; i < R; ++i) {
        uchar* dstRow = dst.ptr<uchar>(2 * i);
        for (int j = 0; j < C; ++j) {
            int oddCol = 2 * j + 1;
            if (oddCol >= 2 * C) continue;

            uchar* mid = dstRow + oddCol * elemSize;
            if (j == C - 1) {
                // sem vizinho a direita: replica o valor da esquerda
                uchar* left = dstRow + (oddCol - 1) * elemSize;
                memcpy(mid, left, elemSize);
            } else {
                uchar* left  = dstRow + (oddCol - 1) * elemSize;
                uchar* right = dstRow + (oddCol + 1) * elemSize;
                for (int e = 0; e < elemSize; ++e) {
                    mid[e] = static_cast<uchar>(lround((left[e] + right[e]) / 2.0));
                }
            }
        }
    }

    // Passo 3: interpolacao vertical em TODAS as colunas (linhas impares),
    // usando as linhas pares ja completamente preenchidas pelo passo 2.
    const int totalBytesPerRow = 2 * C * elemSize;
    for (int i = 0; i < R; ++i) {
        int oddRow = 2 * i + 1;
        if (oddRow >= 2 * R) continue;

        uchar* midRow = dst.ptr<uchar>(oddRow);
        uchar* aboveRow = dst.ptr<uchar>(oddRow - 1);

        if (i == R - 1) {
            // sem linha par seguinte: replica a linha de cima
            memcpy(midRow, aboveRow, totalBytesPerRow);
        } else {
            uchar* belowRow = dst.ptr<uchar>(oddRow + 1);
            for (int x = 0; x < totalBytesPerRow; ++x) {
                midRow[x] = static_cast<uchar>(lround((aboveRow[x] + belowRow[x]) / 2.0));
            }
        }
    }

    return dst;
}


// (9) Rotacao de 90 graus. Troca linhas por colunas (dst.rows = src.cols,
// dst.cols = src.rows) e reposiciona cada pixel conforme o sentido.

Mat rotate90(const Mat& src, bool clockwise) {
    const int rows = src.rows;
    const int cols = src.cols;
    const int elemSize = static_cast<int>(src.elemSize());

    Mat dst(cols, rows, src.type());

    for (int i = 0; i < rows; ++i) {
        const uchar* srcRow = src.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            const uchar* px = srcRow + j * elemSize;

            int di, dj;
            if (clockwise) {
                di = j;
                dj = rows - 1 - i;
            } else {
                di = cols - 1 - j;
                dj = i;
            }

            uchar* dstPx = dst.ptr<uchar>(di) + dj * elemSize;
            memcpy(dstPx, px, elemSize);
        }
    }
    return dst;
}

} // namespace geom
