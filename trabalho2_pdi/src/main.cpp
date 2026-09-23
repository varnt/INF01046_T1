// ============================================================================
// INF01046 - Fundamentos de Processamento de Imagens
// Trabalho 2 - Transformacoes Lineares, Equalizacao e Matching de
// Histograma, Convolucao e Filtragem no Dominio Espacial
//
// Interface 100% dentro das janelas do OpenCV: parametros continuos
// (brilho, contraste, niveis de quantizacao, fatores de zoom, kernel de
// convolucao) sao trackbars; as demais operacoes sao atalhos de teclado.
// NENHUMA leitura de std::cin acontece durante a interacao: o loop
// principal chama cv::waitKey continuamente, entao a janela nunca fica
// "not responding" esperando entrada de texto no console.
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
static const std::string WIN_CONTROLS = "Controles";
static const std::string WIN_HIST     = "Histograma";

// --- Estado global ---
cv::Mat g_original;     // carregada uma vez, nunca modificada
cv::Mat g_base;         // imagem "salva": resultado das ultimas operacoes aplicadas
cv::Mat g_reference1C;  // imagem de referencia (histogram matching), opcional
bool    g_hasReference = false;

// --- Trackbars (parametros continuos) ---
int g_brightness = 255;   // 0..510  -> delta = valor - 255  (-255..255)
int g_contrast   = 100;   // 1..500  -> fator = valor / 100.0 (0.01..5.00)
int g_levels     = 256;   // 2..256  -> niveis de quantizacao
int g_sxTrack    = 20;    // 10..500 -> sx = valor / 10.0 (1.0..50.0)
int g_syTrack    = 20;    // 10..500 -> sy = valor / 10.0 (1.0..50.0)
int g_kernelIdx  = 0;     // 0..7    -> 0 = nenhum; 1..7 = kernels do enunciado

std::vector<conv::Kernel3x3> g_kernels;

static void updateDisplay(int = 0, void* = nullptr) {
    double delta  = g_brightness - 255;
    double factor = g_contrast / 100.0;

    cv::Mat preview = pointops::adjustBrightness(g_base, static_cast<int>(delta));
    preview = pointops::adjustContrast(preview, factor);

    cv::imshow(WIN_RESULT, preview);
}

// "Salva" o efeito atual das trackbars de brilho/contraste na imagem base,
// e as reseta para o ponto neutro. Chamado antes de qualquer operacao de
// tecla (negativo, equalizar, girar, etc.) para que o brilho/contraste
// ajustados na hora nao se percam nem sejam aplicados em dobro.
static void commitPreview() {
    double delta  = g_brightness - 255;
    double factor = g_contrast / 100.0;

    g_base = pointops::adjustBrightness(g_base, static_cast<int>(delta));
    g_base = pointops::adjustContrast(g_base, factor);

    g_brightness = 255;
    g_contrast = 100;
    cv::setTrackbarPos("Brilho", WIN_CONTROLS, g_brightness);
    cv::setTrackbarPos("Contraste", WIN_CONTROLS, g_contrast);
}

static void showHistogram() {
    cv::Mat lum = imgops::toGrayscaleLuminance1C(g_base);
    histo::Hist256 h = histo::computeHistogram(lum);
    cv::Mat histImg = histo::drawHistogram(h);
    cv::namedWindow(WIN_HIST, cv::WINDOW_AUTOSIZE);
    cv::imshow(WIN_HIST, histImg);
}

static void applyConvolution() {
    if (g_kernelIdx <= 0 || static_cast<size_t>(g_kernelIdx) > g_kernels.size()) {
        std::cout << "[convolucao] Selecione um kernel (1-7) na trackbar 'Kernel' antes de apertar 'k'.\n";
        return;
    }
    const conv::Kernel3x3& k = g_kernels[g_kernelIdx - 1];
    std::cout << "[convolucao] Aplicando kernel: " << k.name << "\n";

    if (k.colorAllowed && g_base.channels() == 3) {
        g_base = conv::convolve3x3Color(g_base, k.w, k.addOffset127);
    } else {
        cv::Mat gray1C = imgops::toGrayscaleLuminance1C(g_base);
        cv::Mat result1C = conv::convolve3x3Gray(gray1C, k.w, k.addOffset127);
        cv::merge(std::vector<cv::Mat>{result1C, result1C, result1C}, g_base);
    }
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

static void printHelp() {
    std::cout <<
        "\n================= COMANDOS =================\n"
        "Trackbars (janela 'Controles'):\n"
        "  Brilho      : ajuste em tempo real (-255..255)\n"
        "  Contraste   : ajuste em tempo real (fator 0.01..5.00)\n"
        "  Niveis      : niveis de quantizacao (usado com a tecla 'u')\n"
        "  ZoomSx/ZoomSy: fatores de reducao (usados com a tecla 'z')\n"
        "  Kernel      : 0=nenhum, 1..7=kernel de convolucao (usado com 'k')\n"
        "                1=Gaussiano 2=Laplaciano 3=PassaAltasGenerico\n"
        "                4=PrewittHx 5=PrewittHy 6=SobelHx 7=SobelHy\n"
        "\n"
        "Teclado (com a janela 'Resultado' em foco):\n"
        "  n - negativo\n"
        "  g - converter para tons de cinza (luminancia)\n"
        "  e - equalizar histograma (cinza / cor via luminancia)\n"
        "  l - equalizar histograma em L*a*b* [extra]\n"
        "  u - quantizar tons (usa a trackbar 'Niveis')\n"
        "  x - espelhar horizontal\n"
        "  y - espelhar vertical\n"
        "  z - zoom out / reduzir (usa as trackbars 'ZoomSx'/'ZoomSy')\n"
        "  Z - zoom in 2x2 (shift+z)\n"
        "  r - rotacionar 90 horario\n"
        "  R - rotacionar 90 anti-horario (shift+r)\n"
        "  k - aplicar convolucao com o kernel selecionado na trackbar\n"
        "  m - histogram matching com a imagem de referencia (2o argumento)\n"
        "  h - mostrar histograma (janela separada)\n"
        "  o - restaurar imagem original\n"
        "  s - salvar imagem atual (gera resultado_NNN.jpg)\n"
        "  ? - mostrar esta ajuda novamente\n"
        "  q / ESC - sair\n"
        "==============================================\n" << std::endl;
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
                      << argv[2] << "'. Histogram matching ('m') ficara indisponivel." << std::endl;
        } else {
            g_reference1C = imgops::toGrayscaleLuminance1C(refColor);
            g_hasReference = true;
        }
    }

    g_kernels = conv::builtinKernels();

    cv::namedWindow(WIN_ORIGINAL, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(WIN_RESULT, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(WIN_CONTROLS, cv::WINDOW_NORMAL);
    cv::resizeWindow(WIN_CONTROLS, 420, 220);

    cv::imshow(WIN_ORIGINAL, g_original);

    cv::createTrackbar("Brilho", WIN_CONTROLS, &g_brightness, 510, updateDisplay);
    cv::createTrackbar("Contraste", WIN_CONTROLS, &g_contrast, 500, updateDisplay);
    cv::setTrackbarMin("Contraste", WIN_CONTROLS, 1);
    cv::createTrackbar("Niveis", WIN_CONTROLS, &g_levels, 256, nullptr);
    cv::setTrackbarMin("Niveis", WIN_CONTROLS, 2);
    cv::createTrackbar("ZoomSx", WIN_CONTROLS, &g_sxTrack, 500, nullptr);
    cv::setTrackbarMin("ZoomSx", WIN_CONTROLS, 10);
    cv::createTrackbar("ZoomSy", WIN_CONTROLS, &g_syTrack, 500, nullptr);
    cv::setTrackbarMin("ZoomSy", WIN_CONTROLS, 10);
    cv::createTrackbar("Kernel", WIN_CONTROLS, &g_kernelIdx, 7, nullptr);
    

    updateDisplay();
    printHelp();

    bool running = true;
    while (running) {
        int key = cv::waitKey(30) & 0xFF; // mantem a fila de eventos da GUI sempre viva

        switch (key) {
            case 'n':
                commitPreview();
                g_base = pointops::negative(g_base);
                updateDisplay();
                break;
            case 'g':
                commitPreview();
                g_base = imgops::toGrayscaleLuminance(g_base);
                updateDisplay();
                break;
            case 'e':
                commitPreview();
                g_base = histo::equalizeColorViaLuminance(g_base);
                updateDisplay();
                break;
            case 'l':
                commitPreview();
                g_base = labops::equalizeLab(g_base);
                updateDisplay();
                break;
            case 'u': {
                commitPreview();
                cv::Mat gray = imgops::toGrayscaleLuminance(g_base);
                g_base = imgops::quantize(gray, g_levels);
                updateDisplay();
                break;
            }
            case 'x':
                commitPreview();
                g_base = imgops::mirror(g_base, true, false);
                updateDisplay();
                break;
            case 'y':
                commitPreview();
                g_base = imgops::mirror(g_base, false, true);
                updateDisplay();
                break;
            case 'z': {
                commitPreview();
                double sx = g_sxTrack / 10.0;
                double sy = g_syTrack / 10.0;
                g_base = geom::zoomOut(g_base, sx, sy);
                updateDisplay();
                break;
            }
            case 'Z':
                commitPreview();
                g_base = geom::zoomIn2x(g_base);
                updateDisplay();
                break;
            case 'r':
                commitPreview();
                g_base = geom::rotate90(g_base, true);
                updateDisplay();
                break;
            case 'R':
                commitPreview();
                g_base = geom::rotate90(g_base, false);
                updateDisplay();
                break;
            case 'k':
                commitPreview();
                applyConvolution();
                updateDisplay();
                break;
            case 'm': {
                if (!g_hasReference) {
                    std::cout << "[matching] Nenhuma imagem de referencia foi passada "
                                 "como 2o argumento na linha de comando.\n";
                    break;
                }
                commitPreview();
                cv::Mat srcGray = imgops::toGrayscaleLuminance1C(g_base);
                cv::Mat matched1C = histo::histogramMatching(srcGray, g_reference1C);
                cv::merge(std::vector<cv::Mat>{matched1C, matched1C, matched1C}, g_base);
                updateDisplay();
                break;
            }
            case 'h':
                showHistogram();
                break;
            case 'o':
                g_base = g_original.clone();
                g_brightness = 255;
                g_contrast = 100;
                cv::setTrackbarPos("Brilho", WIN_CONTROLS, g_brightness);
                cv::setTrackbarPos("Contraste", WIN_CONTROLS, g_contrast);
                updateDisplay();
                break;
            case 's':
                saveCurrent();
                break;
            case '?':
                printHelp();
                break;
            case 'q':
            case 27: // ESC
                running = false;
                break;
            default:
                break; // tecla sem funcao (ou nenhuma tecla pressionada nesse ciclo)
        }
    }

    return 0;
}
