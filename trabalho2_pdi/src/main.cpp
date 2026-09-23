// ============================================================================
// INF01046 - Fundamentos de Processamento de Imagens
// Trabalho 2 - Transformacoes Lineares, Equalizacao e Matching de
// Histograma, Convolucao e Filtragem no Dominio Espacial
//
// Interface 100% dentro do OpenCV: os parametros continuos (brilho,
// contraste, niveis de quantizacao, fatores de zoom) sao trackbars, e
// TODAS as demais operacoes sao BOTOES (cv::createButton, requer OpenCV
// compilado com suporte a Qt). Trackbars e botoes sao criados com o
// "painel de controles" automatico do Qt (janela auxiliar que o OpenCV
// abre sozinho na primeira trackbar/botao criado sem estar associado a
// uma janela de imagem) - e por isso nao aparecem dentro da janela
// "Resultado" nem "Original", mas sim numa janela separada dedicada aos
// controles, exatamente como pedido.
//
// Nenhuma leitura de std::cin acontece durante o uso: o loop principal
// so chama cv::waitKey em loop para manter a fila de eventos da GUI viva
// (os cliques nos botoes chegam via callback, nao pelo teclado).
//
// Uso:
//   ./trabalho2 <imagem_entrada> [imagem_referencia_para_histogram_matching]
// ============================================================================

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

static const std::string WIN_ORIGINAL = "Original";
static const std::string WIN_RESULT   = "Resultado";
static const std::string WIN_HIST     = "Histograma";

// --- Estado global ---
cv::Mat g_original;     // carregada uma vez, nunca modificada
cv::Mat g_base;         // imagem "salva": resultado das ultimas operacoes aplicadas
cv::Mat g_reference1C;  // imagem de referencia (histogram matching), opcional
bool    g_hasReference = false;
bool    g_quit = false;

// --- Trackbars (parametros continuos) ---
int g_brightness = 255;   // 0..510  -> delta = valor - 255  (-255..255)
int g_contrast   = 100;   // 1..500  -> fator = valor / 100.0 (0.01..5.00)
int g_levels     = 256;   // 2..256  -> niveis de quantizacao
int g_sxTrack    = 20;    // 10..500 -> sx = valor / 10.0 (1.0..50.0)
int g_syTrack    = 20;    // 10..500 -> sy = valor / 10.0 (1.0..50.0)

std::vector<conv::Kernel3x3> g_kernels;
static const int kKernelIndex[7] = {1, 2, 3, 4, 5, 6, 7}; // userdata dos botoes de kernel

static void updateDisplay() {
    double delta  = g_brightness - 255;
    double factor = g_contrast / 100.0;

    cv::Mat preview = pointops::adjustBrightness(g_base, static_cast<int>(delta));
    preview = pointops::adjustContrast(preview, factor);

    cv::imshow(WIN_RESULT, preview);
    cv::waitKey(1);
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
    cv::setTrackbarPos("Brilho", "", g_brightness);
    cv::setTrackbarPos("Contraste", "", g_contrast);
}

static void showHistogram() {
    cv::Mat lum = imgops::toGrayscaleLuminance1C(g_base);
    histo::Hist256 h = histo::computeHistogram(lum);
    cv::Mat histImg = histo::drawHistogram(h);
    cv::namedWindow(WIN_HIST, cv::WINDOW_AUTOSIZE);
    cv::imshow(WIN_HIST, histImg);
    cv::waitKey(1);
}

static void applyKernel(int kernelNumber) {
    if (kernelNumber < 1 || static_cast<size_t>(kernelNumber) > g_kernels.size()) return;

    commitPreview();
    const conv::Kernel3x3& k = g_kernels[kernelNumber - 1];
    std::cout << "[convolucao] Aplicando kernel: " << k.name << std::endl;

    if (k.colorAllowed && g_base.channels() == 3) {
        g_base = conv::convolve3x3Color(g_base, k.w, k.addOffset127);
    } else {
        cv::Mat gray1C = imgops::toGrayscaleLuminance1C(g_base);
        cv::Mat result1C = conv::convolve3x3Gray(gray1C, k.w, k.addOffset127);
        cv::merge(std::vector<cv::Mat>{result1C, result1C, result1C}, g_base);
    }
    updateDisplay();
}

static void saveCurrent() {
    static int counter = 1;
    std::ostringstream oss;
    oss << "resultado_" << std::setw(3) << std::setfill('0') << counter++ << ".jpg";
    std::string path = oss.str();

    double delta  = g_brightness - 255;
    double factor = g_contrast / 100.0;
    cv::Mat toSave = pointops::adjustBrightness(g_base, static_cast<int>(delta));
    toSave = pointops::adjustContrast(toSave, factor);

    std::vector<int> jpegParams = { cv::IMWRITE_JPEG_QUALITY, 95 };
    bool ok = cv::imwrite(path, toSave, jpegParams);
    std::cout << (ok ? "[salvar] Imagem salva em: " + path
                      : std::string("[salvar] ERRO ao salvar a imagem."))
              << std::endl;
}

// ---------------------------------------------------------------------
// Callbacks dos botoes (cv::ButtonCallback: void(int state, void* userdata))
// ---------------------------------------------------------------------
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
    cv::Mat gray = imgops::toGrayscaleLuminance(g_base);
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
        std::cout << "[matching] Nenhuma imagem de referencia foi passada "
                     "como 2o argumento na linha de comando." << std::endl;
        return;
    }
    commitPreview();
    cv::Mat srcGray = imgops::toGrayscaleLuminance1C(g_base);
    cv::Mat matched1C = histo::histogramMatching(srcGray, g_reference1C);
    cv::merge(std::vector<cv::Mat>{matched1C, matched1C, matched1C}, g_base);
    updateDisplay();
}
static void onMostrarHistograma(int, void*) {
    showHistogram();
}
static void onRestaurarOriginal(int, void*) {
    g_base = g_original.clone();
    g_brightness = 255;
    g_contrast = 100;
    cv::setTrackbarPos("Brilho", "", g_brightness);
    cv::setTrackbarPos("Contraste", "", g_contrast);
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
        std::cerr << "Uso: " << argv[0] << " <imagem_entrada> [imagem_referencia]" << std::endl;
        return 1;
    }

    g_original = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (g_original.empty()) {
        std::cerr << "Erro: nao foi possivel abrir a imagem '" << argv[1] << "'." << std::endl;
        return 1;
    }
    g_base = g_original.clone();

    if (argc == 3) {
        cv::Mat refColor = cv::imread(argv[2], cv::IMREAD_COLOR);
        if (refColor.empty()) {
            std::cerr << "Aviso: nao foi possivel abrir a imagem de referencia '"
                      << argv[2] << "'. Histogram matching ficara indisponivel." << std::endl;
        } else {
            g_reference1C = imgops::toGrayscaleLuminance1C(refColor);
            g_hasReference = true;
        }
    }

    g_kernels = conv::builtinKernels();

    // As janelas de imagem precisam existir antes de criar trackbars/botoes
    // no painel de controles do Qt.
    cv::namedWindow(WIN_ORIGINAL, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(WIN_RESULT, cv::WINDOW_AUTOSIZE);
    cv::imshow(WIN_ORIGINAL, g_original);

    // --- Trackbars: winname = "" -> vao para o painel de controles do Qt,
    // junto com os botoes criados a seguir (mesma janela de controles). ---
    cv::createTrackbar("Brilho", "", &g_brightness, 510, onTrackbarChange);
    cv::createTrackbar("Contraste", "", &g_contrast, 500, onTrackbarChange);
    cv::setTrackbarMin("Contraste", "", 1);
    cv::createTrackbar("Niveis (quantizacao)", "", &g_levels, 256, nullptr);
    cv::setTrackbarMin("Niveis (quantizacao)", "", 2);
    cv::createTrackbar("ZoomSx (x0.1)", "", &g_sxTrack, 500, nullptr);
    cv::setTrackbarMin("ZoomSx (x0.1)", "", 10);
    cv::createTrackbar("ZoomSy (x0.1)", "", &g_syTrack, 500, nullptr);
    cv::setTrackbarMin("ZoomSy (x0.1)", "", 10);

    // --- Botoes: uma acao por botao, todos no mesmo painel de controles. ---
    cv::createButton("Negativo", onNegativo, nullptr, cv::QT_PUSH_BUTTON);
    cv::createButton("Tons de Cinza", onTonsDeCinza, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Equalizar Histograma", onEqualizar, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Equalizar Histograma (Lab) [extra]", onEqualizarLab, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Quantizar Tons", onQuantizar, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Espelhar Horizontal", onEspelharH, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Espelhar Vertical", onEspelharV, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Zoom Out (Reduzir)", onZoomOut, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Zoom In 2x2 (Ampliar)", onZoomIn, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Rotacionar 90 Horario", onRotacionarHorario, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Rotacionar 90 Anti-horario", onRotacionarAntiHorario, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);

    // Um botao por kernel 
    cv::createButton("Kernel: Gaussiano (passa-baixas)", onKernelButton,
                      const_cast<int*>(&kKernelIndex[0]), cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Kernel: Laplaciano (passa-altas)", onKernelButton,
                      const_cast<int*>(&kKernelIndex[1]), cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Kernel: Passa-Altas Generico", onKernelButton,
                      const_cast<int*>(&kKernelIndex[2]), cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Kernel: Prewitt Hx", onKernelButton,
                      const_cast<int*>(&kKernelIndex[3]), cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Kernel: Prewitt Hy", onKernelButton,
                      const_cast<int*>(&kKernelIndex[4]), cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Kernel: Sobel Hx", onKernelButton,
                      const_cast<int*>(&kKernelIndex[5]), cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Kernel: Sobel Hy", onKernelButton,
                      const_cast<int*>(&kKernelIndex[6]), cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);

    cv::createButton("Histogram Matching (2o argumento)", onHistogramMatching, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Mostrar Histograma", onMostrarHistograma, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Restaurar Original", onRestaurarOriginal, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Salvar Imagem (JPEG)", onSalvar, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);
    cv::createButton("Sair", onSair, nullptr, cv::QT_PUSH_BUTTON | cv::QT_NEW_BUTTONBAR);

    std::cout << "Todos os controles (sliders e botoes) estao na janela "
                 "'Control Panel' aberta pelo OpenCV/Qt.\n"
                 "Se ela nao aparecer em primeiro plano, procure na barra de "
                 "tarefas ou clique no icone de engrenagem no topo da janela "
                 "'Resultado'." << std::endl;

    updateDisplay();

    while (!g_quit) {
        int key = cv::waitKey(30) & 0xFF; // mantem a fila de eventos da GUI viva
        if (key == 'q' || key == 27) {    // ESC continua funcionando como atalho de saida
            g_quit = true;
        }
    }

    return 0;
}
