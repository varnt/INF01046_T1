

#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace imgops {

    // --- Declarações ---
    Mat mirror(const Mat& src, bool horizontal, bool vertical);
    Mat toGrayscaleLuminance(const Mat& src);
    Mat quantize(const Mat& grayImg, int n);

    // --- Implementações da parte 2 ---
    
    // (a) Espelhamento horizontal e vertical
    Mat mirror(const Mat& src, bool horizontal, bool vertical) {
        Mat dst = src.clone();
        const int rows = dst.rows;
        const int cols = dst.cols;
        const size_t elemSize = dst.elemSize();

        if (vertical) {
            vector<uchar> temp(dst.step);
            for (int i = 0; i < rows / 2; ++i) {
                uchar* rowTop = dst.ptr<uchar>(i);
                uchar* rowBot = dst.ptr<uchar>(rows - 1 - i);
                memcpy(temp.data(), rowTop, dst.step);
                memcpy(rowTop, rowBot, dst.step);
                memcpy(rowBot, temp.data(), dst.step);
            }
        }

        if (horizontal) {
            vector<uchar> temp(elemSize);
            for (int i = 0; i < rows; ++i) {
                uchar* rowPtr = dst.ptr<uchar>(i);
                for (int j = 0; j < cols / 2; ++j) {
                    uchar* pixelLeft  = rowPtr + j * elemSize;
                    uchar* pixelRight = rowPtr + (cols - 1 - j) * elemSize;
                    memcpy(temp.data(), pixelLeft, elemSize);
                    memcpy(pixelLeft, pixelRight, elemSize);
                    memcpy(pixelRight, temp.data(), elemSize);
                }
            }
        }
        return dst;
    }

    // (b) Conversao para tons de cinza (luminancia)
    Mat toGrayscaleLuminance(const Mat& src) {
        CV_Assert(src.channels() == 3);
        Mat dst(src.rows, src.cols, src.type());

        for (int i = 0; i < src.rows; ++i) {
            const Vec3b* srcRow = src.ptr<Vec3b>(i);
            Vec3b* dstRow = dst.ptr<Vec3b>(i);

            for (int j = 0; j < src.cols; ++j) {
                double B = srcRow[j][0];
                double G = srcRow[j][1];
                double R = srcRow[j][2];

                double L = 0.299 * R + 0.587 * G + 0.114 * B;
                int Li = static_cast<int>(lround(L));
                Li = min(255, max(0, Li));

                uchar Lc = static_cast<uchar>(Li);
                dstRow[j] = Vec3b(Lc, Lc, Lc);
            }
        }
        return dst;
    }

    // (c) Quantizacao de tons
    Mat quantize(const Mat& grayImg, int n) {
        CV_Assert(grayImg.channels() == 3);
        if (n <= 0) return grayImg.clone();

        int t1 = 255;
        int t2 = 0;
        for (int i = 0; i < grayImg.rows; ++i) {
            const Vec3b* row = grayImg.ptr<Vec3b>(i);
            for (int j = 0; j < grayImg.cols; ++j) {
                int v = row[j][0];
                if (v < t1) t1 = v;
                if (v > t2) t2 = v;
            }
        }

        int tam_int = t2 - t1 + 1;
        if (n >= tam_int) return grayImg.clone();

        double tb = static_cast<double>(tam_int) / static_cast<double>(n);
        Mat dst(grayImg.rows, grayImg.cols, grayImg.type());

        for (int i = 0; i < grayImg.rows; ++i) {
            const Vec3b* srcRow = grayImg.ptr<Vec3b>(i);
            Vec3b* dstRow = dst.ptr<Vec3b>(i);

            for (int j = 0; j < grayImg.cols; ++j) {
                double t_orig = srcRow[j][0];
                int binIdx = static_cast<int>(floor((t_orig - (t1 - 0.5)) / tb));
                if (binIdx < 0)      binIdx = 0;
                if (binIdx >= n)     binIdx = n - 1;

                double binStart = (t1 - 0.5) + binIdx * tb;
                double center   = binStart + tb / 2.0;

                int q = static_cast<int>(lround(center));
                q = min(255, max(0, q));

                uchar qc = static_cast<uchar>(q);
                dstRow[j] = Vec3b(qc, qc, qc);
            }
        }
        return dst;
    }

} // namespace imgops


// ============================================================================
// INTERFACE E INTERAÇÃO (Originalmente main.cpp)
// ============================================================================

Mat g_original;   
Mat g_result;     

int g_mirrorH   = 0;  
int g_mirrorV   = 0;  
int g_grayscale = 0;  
int g_levels    = 256; 

const string WIN_CONTROLS = "Controles";
const string WIN_DISPLAY  = "Original  |  Resultado";

void updateDisplay(int = 0, void* = nullptr) {
    Mat step = g_original;

    if (g_mirrorH || g_mirrorV) {
        step = imgops::mirror(step, g_mirrorH != 0, g_mirrorV != 0);
    }

    if (g_grayscale) {
        step = imgops::toGrayscaleLuminance(step);
        int n = max(2, g_levels);
        step = imgops::quantize(step, n);
    }

    g_result = step;

    Mat leftView = g_original;
    Mat rightView = g_result;
    Mat combined;
    hconcat(leftView, rightView, combined);

    line(combined,
             Point(leftView.cols, 0),
             Point(leftView.cols, combined.rows),
             Scalar(0, 0, 255), 2);

    imshow(WIN_DISPLAY, combined);
}

void onMirrorH(int, void*)   { updateDisplay(); }
void onMirrorV(int, void*)   { updateDisplay(); }
void onGrayscale(int, void*) { updateDisplay(); }
void onLevels(int, void*)    { updateDisplay(); }

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Uso: " << argv[0] << " <imagem_entrada>" << endl;
        return 1;
    }

    string inputPath = argv[1];

    g_original = imread(inputPath, IMREAD_COLOR);
    if (g_original.empty()) {
        cerr << "Erro: nao foi possivel abrir a imagem '" << inputPath << "'." << endl;
        return 1;
    }
    g_result = g_original.clone();

    namedWindow(WIN_DISPLAY, WINDOW_AUTOSIZE);
    namedWindow(WIN_CONTROLS, WINDOW_NORMAL);
    resizeWindow(WIN_CONTROLS, 420, 150);

    createTrackbar("Espelho H",  WIN_CONTROLS, &g_mirrorH,   1, onMirrorH);
    createTrackbar("Espelho V",  WIN_CONTROLS, &g_mirrorV,   1, onMirrorV);
    createTrackbar("Tons Cinza", WIN_CONTROLS, &g_grayscale, 1, onGrayscale);
    createTrackbar("Niveis",     WIN_CONTROLS, &g_levels,    256, onLevels);
    setTrackbarMin("Niveis", WIN_CONTROLS, 2);

    updateDisplay();

    cout << "Comandos:\n"
              << "  's' -> salvar imagem resultante em JPEG automaticamente (_out)\n"
              << "  'q' ou ESC -> sair\n";

    while (true) {
        int key = waitKey(30) & 0xFF;

        if (key == 'q' || key == 27) { // ESC
            break;
        }
        if (key == 's' || key == 'S') {
            // Lógica automática de geração de nome "filename_out"
            string outPath;
            size_t lastDot = inputPath.find_last_of(".");
            
            if (lastDot != string::npos) {
                outPath = inputPath.substr(0, lastDot) + "_out" + inputPath.substr(lastDot);
            } else {
                outPath = inputPath + "_out.jpg"; // Fallback caso não tenha extensão
            }

            vector<int> jpegParams = { IMWRITE_JPEG_QUALITY, 95 };
            bool ok = imwrite(outPath, g_result, jpegParams);

            if (ok) {
                cout << "Imagem salva automaticamente em: " << outPath << endl;
            } else {
                cerr << "Erro ao salvar a imagem em: " << outPath << endl;
            }
        }
    }

    return 0;
}