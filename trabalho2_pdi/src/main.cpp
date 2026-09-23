// INF01046 - Fundamentos de Processamento de Imagens
// Trabalho 2 - Transformacoes Lineares, Equalizacao e Matching de
// Histograma, Convolucao e Filtragem no Dominio Espacial
//
// Uso:
//   ./trabalho2 <imagem_entrada> [imagem_referencia_para_histogram_matching]


#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "ImageOps.h"
#include "PointOps.h"
#include "Histogram.h"
#include "LabOps.h"
#include "Geometry.h"
#include "Convolution.h"

using namespace std;
using namespace cv;

static const string WIN_ORIGINAL = "Original";
static const string WIN_RESULT   = "Resultado";
static const string WIN_HIST     = "Histograma";

//Estado global
Mat g_original;     // carregada uma vez, nunca modificada
Mat g_base;         // imagem "salva": resultado das ultimas operacoes aplicadas
Mat g_reference1C;  // imagem de referencia (histogram matching), opcional
bool    g_hasReference = false;
bool    g_quit = false;

//Trackbars (parametros continuos)
int g_brightness = 255;   // 0..510  -> delta = valor - 255  (-255..255)
int g_contrast   = 100;   // 1..500  -> fator = valor / 100.0 (0.01..5.00)
int g_levels     = 256;   // 2..256  -> niveis de quantizacao
int g_sxTrack    = 20;    // 10..500 -> sx = valor / 10.0 (1.0..50.0)
int g_syTrack    = 20;    // 10..500 -> sy = valor / 10.0 (1.0..50.0)

vector<conv::Kernel3x3> g_kernels;
static const int kKernelIndex[7] = {1, 2, 3, 4, 5, 6, 7}; // userdata dos botoes de kernel

static void updateDisplay() {
    double delta  = g_brightness - 255;
    double factor = g_contrast / 100.0;

    Mat preview = pointops::adjustBrightness(g_base, static_cast<int>(delta));
    preview = pointops::adjustContrast(preview, factor);

    imshow(WIN_RESULT, preview);
    waitKey(1);
}

static void onTrackbarChange(int, void*) { updateDisplay(); }

// "Salva" o efeito atual das trackbars de brilho/contraste na imagem base,
// e as reseta para o ponto neutro. Chamado antes de qualquer botao de
// operacao, para que o brilho/contraste ajustados na hora nao se percam
// nem sejam aplicados em dobro.
static void commitPreview() {
    double delta  = g_brightness - 255;
    double factor = g_contrast / 100.0;

    g_base = pointops::adjustBrightness(g_base, static_cast<int>(delta));
    g_base = pointops::adjustContrast(g_base, factor);

    g_brightness = 255;
    g_contrast = 100;
    setTrackbarPos("Brilho", "", g_brightness);
    setTrackbarPos("Contraste", "", g_contrast);
}

static void showHistogram() {
    Mat lum = imgops::toGrayscaleLuminance1C(g_base);
    histo::Hist256 h = histo::computeHistogram(lum);
    Mat histImg = histo::drawHistogram(h);
    namedWindow(WIN_HIST, WINDOW_AUTOSIZE);
    imshow(WIN_HIST, histImg);
    waitKey(1);
}

static void applyKernel(int kernelNumber) {
    if (kernelNumber < 1 || static_cast<size_t>(kernelNumber) > g_kernels.size()) return;

    commitPreview();
    const conv::Kernel3x3& k = g_kernels[kernelNumber - 1];
    cout << "[convolucao] Aplicando kernel: " << k.name << endl;

    if (k.colorAllowed && g_base.channels() == 3) {
        g_base = conv::convolve3x3Color(g_base, k.w, k.addOffset127);
    } else {
        Mat gray1C = imgops::toGrayscaleLuminance1C(g_base);
        Mat result1C = conv::convolve3x3Gray(gray1C, k.w, k.addOffset127);
        merge(vector<Mat>{result1C, result1C, result1C}, g_base);
    }
    updateDisplay();
}

static void saveCurrent() {
    static int counter = 1;
    ostringstream oss;
    oss << "resultado_" << setw(3) << setfill('0') << counter++ << ".jpg";
    string path = oss.str();

    double delta  = g_brightness - 255;
    double factor = g_contrast / 100.0;
    Mat toSave = pointops::adjustBrightness(g_base, static_cast<int>(delta));
    toSave = pointops::adjustContrast(toSave, factor);

    vector<int> jpegParams = { IMWRITE_JPEG_QUALITY, 95 };
    bool ok = imwrite(path, toSave, jpegParams);
    cout << (ok ? "[salvar] Imagem salva em: " + path
                      : string("[salvar] ERRO ao salvar a imagem."))
              << endl;
}


// Callbacks dos botoes (ButtonCallback: void(int state, void* userdata))

static void onNegativo(int, void*) {
    commitPreview();
    g_base = pointops::negative(g_base);
    updateDisplay();
}
static void onTonsDeCinza(int, void*) {
    commitPreview();
    g_base = imgops::toGrayscaleLuminance(g_base);
    updateDisplay();
}
static void onEqualizar(int, void*) {
    commitPreview();
    g_base = histo::equalizeColorViaLuminance(g_base);
    updateDisplay();
}
static void onEqualizarLab(int, void*) {
    commitPreview();
    g_base = labops::equalizeLab(g_base);
    updateDisplay();
}
static void onQuantizar(int, void*) {
    commitPreview();
    Mat gray = imgops::toGrayscaleLuminance(g_base);
    g_base = imgops::quantize(gray, g_levels);
    updateDisplay();
}
static void onEspelharH(int, void*) {
    commitPreview();
    g_base = imgops::mirror(g_base, true, false);
    updateDisplay();
}
static void onEspelharV(int, void*) {
    commitPreview();
    g_base = imgops::mirror(g_base, false, true);
    updateDisplay();
}
static void onZoomOut(int, void*) {
    commitPreview();
    double sx = g_sxTrack / 10.0;
    double sy = g_syTrack / 10.0;
    g_base = geom::zoomOut(g_base, sx, sy);
    updateDisplay();
}
static void onZoomIn(int, void*) {
    commitPreview();
    g_base = geom::zoomIn2x(g_base);
    updateDisplay();
}
static void onRotacionarHorario(int, void*) {
    commitPreview();
    g_base = geom::rotate90(g_base, true);
    updateDisplay();
}
static void onRotacionarAntiHorario(int, void*) {
    commitPreview();
    g_base = geom::rotate90(g_base, false);
    updateDisplay();
}
static void onHistogramMatching(int, void*) {
    if (!g_hasReference) {
        cout << "[matching] Nenhuma imagem de referencia foi passada "
                     "como 2o argumento na linha de comando." << endl;
        return;
    }
    commitPreview();
    Mat srcGray = imgops::toGrayscaleLuminance1C(g_base);
    Mat matched1C = histo::histogramMatching(srcGray, g_reference1C);
    merge(vector<Mat>{matched1C, matched1C, matched1C}, g_base);
    updateDisplay();
}
static void onMostrarHistograma(int, void*) {
    showHistogram();
}
static void onRestaurarOriginal(int, void*) {
    g_base = g_original.clone();
    g_brightness = 255;
    g_contrast = 100;
    setTrackbarPos("Brilho", "", g_brightness);
    setTrackbarPos("Contraste", "", g_contrast);
    updateDisplay();
}
static void onSalvar(int, void*) {
    commitPreview();
    saveCurrent();
    updateDisplay();
}
static void onSair(int, void*) {
    g_quit = true;
}
static void onKernelButton(int, void* userdata) {
    int kernelNumber = *static_cast<const int*>(userdata);
    applyKernel(kernelNumber);
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        cerr << "Uso: " << argv[0] << " <imagem_entrada> [imagem_referencia]" << endl;
        return 1;
    }

    g_original = imread(argv[1], IMREAD_COLOR);
    if (g_original.empty()) {
        cerr << "Erro: nao foi possivel abrir a imagem '" << argv[1] << "'." << endl;
        return 1;
    }
    g_base = g_original.clone();

    if (argc == 3) {
        Mat refColor = imread(argv[2], IMREAD_COLOR);
        if (refColor.empty()) {
            cerr << "Aviso: nao foi possivel abrir a imagem de referencia '"
                      << argv[2] << "'. Histogram matching ficara indisponivel." << endl;
        } else {
            g_reference1C = imgops::toGrayscaleLuminance1C(refColor);
            g_hasReference = true;
        }
    }

    g_kernels = conv::builtinKernels();

    // As janelas de imagem precisam existir antes de criar trackbars/botoes
    // no painel de controles do Qt.
    namedWindow(WIN_ORIGINAL, WINDOW_AUTOSIZE);
    namedWindow(WIN_RESULT, WINDOW_AUTOSIZE);
    imshow(WIN_ORIGINAL, g_original);

    //Trackbars: winname = "" -> vao para o painel de controles do Qt,
    // junto com os botoes criados a seguir (mesma janela de controles).
    createTrackbar("Brilho", "", &g_brightness, 510, onTrackbarChange);
    createTrackbar("Contraste", "", &g_contrast, 500, onTrackbarChange);
    setTrackbarMin("Contraste", "", 1);
    createTrackbar("Niveis (quantizacao)", "", &g_levels, 256, nullptr);
    setTrackbarMin("Niveis (quantizacao)", "", 2);
    createTrackbar("ZoomSx (x0.1)", "", &g_sxTrack, 500, nullptr);
    setTrackbarMin("ZoomSx (x0.1)", "", 10);
    createTrackbar("ZoomSy (x0.1)", "", &g_syTrack, 500, nullptr);
    setTrackbarMin("ZoomSy (x0.1)", "", 10);

    //Botoes: uma acao por botao, todos no mesmo painel de controles.
    createButton("Negativo", onNegativo, nullptr, QT_PUSH_BUTTON);
    createButton("Tons de Cinza", onTonsDeCinza, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Equalizar Histograma", onEqualizar, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Equalizar Histograma (Lab) [extra]", onEqualizarLab, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Quantizar Tons", onQuantizar, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Espelhar Horizontal", onEspelharH, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Espelhar Vertical", onEspelharV, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Zoom Out (Reduzir)", onZoomOut, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Zoom In 2x2 (Ampliar)", onZoomIn, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Rotacionar 90 Horario", onRotacionarHorario, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Rotacionar 90 Anti-horario", onRotacionarAntiHorario, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);

    // Um botao por kernel 
    createButton("Kernel: Gaussiano (passa-baixas)", onKernelButton,
                      const_cast<int*>(&kKernelIndex[0]), QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Kernel: Laplaciano (passa-altas)", onKernelButton,
                      const_cast<int*>(&kKernelIndex[1]), QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Kernel: Passa-Altas Generico", onKernelButton,
                      const_cast<int*>(&kKernelIndex[2]), QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Kernel: Prewitt Hx", onKernelButton,
                      const_cast<int*>(&kKernelIndex[3]), QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Kernel: Prewitt Hy", onKernelButton,
                      const_cast<int*>(&kKernelIndex[4]), QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Kernel: Sobel Hx", onKernelButton,
                      const_cast<int*>(&kKernelIndex[5]), QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Kernel: Sobel Hy", onKernelButton,
                      const_cast<int*>(&kKernelIndex[6]), QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);

    createButton("Histogram Matching (2o argumento)", onHistogramMatching, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Mostrar Histograma", onMostrarHistograma, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Restaurar Original", onRestaurarOriginal, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Salvar Imagem (JPEG)", onSalvar, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);
    createButton("Sair", onSair, nullptr, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR);

    cout << "Todos os controles (sliders e botoes) estao na janela "
                 "'Control Panel' aberta pelo OpenCV/Qt.\n"
                 "Se ela nao aparecer em primeiro plano, procure na barra de "
                 "tarefas ou clique no icone de engrenagem no topo da janela "
                 "'Resultado'." << endl;

    updateDisplay();

    while (!g_quit) {
        int key = waitKey(30) & 0xFF; // mantem a fila de eventos da GUI viva
        if (key == 'q' || key == 27) {    // ESC continua funcionando como atalho de saida
            g_quit = true;
        }
    }

    return 0;
}
