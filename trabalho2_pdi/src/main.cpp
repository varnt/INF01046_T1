// ============================================================================
// INF01046 - Fundamentos de Processamento de Imagens
// Trabalho 2 - Transformacoes Lineares, Equalizacao e Matching de
// Histograma, Convolucao e Filtragem no Dominio Espacial
//
// Integra as operacoes implementadas em ImageOps (T1), PointOps, Histogram,
// LabOps (extra), Geometry e Convolution atraves de um menu no console.
// O OpenCV (highgui) e usado APENAS para leitura/gravacao de arquivos e
// gerencia de janelas, conforme permitido pelo enunciado.
//
// Uso:
//   ./trabalho2 <imagem_entrada>
// ============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include "ImageOps.h"
#include "PointOps.h"
#include "Histogram.h"
#include "LabOps.h"
#include "Geometry.h"
#include "Convolution.h"

static const std::string WIN_ORIGINAL = "Original";
static const std::string WIN_RESULT   = "Resultado";
static const std::string WIN_HIST     = "Histograma";

cv::Mat g_original;   // imagem original, carregada uma unica vez, nunca modificada
cv::Mat g_current;    // imagem "de trabalho": cada operacao parte dela e a substitui

// Atualiza a janela de resultado e processa a fila de eventos do OpenCV.
static void refresh() {
    cv::imshow(WIN_RESULT, g_current);
    cv::waitKey(1);
}

static void printMenu() {
    std::cout << "\n==================== MENU ====================\n"
              << " 1  - Ajustar brilho\n"
              << " 2  - Ajustar contraste\n"
              << " 3  - Calcular negativo\n"
              << " 4  - Mostrar histograma (luminancia da imagem atual)\n"
              << " 5  - Equalizar histograma (cinza / cor via luminancia)\n"
              << " 6  - Equalizar histograma em L*a*b* [extra]\n"
              << " 7  - Histogram matching com outra imagem\n"
              << " 8  - Converter para tons de cinza (luminancia)\n"
              << " 9  - Quantizar tons (converte p/ cinza automaticamente)\n"
              << " 10 - Espelhar horizontal\n"
              << " 11 - Espelhar vertical\n"
              << " 12 - Zoom out (reduzir)\n"
              << " 13 - Zoom in 2x2 (ampliar)\n"
              << " 14 - Rotacionar 90 (horario)\n"
              << " 15 - Rotacionar 90 (anti-horario)\n"
              << " 16 - Aplicar convolucao 3x3 (escolher kernel)\n"
              << " 17 - Restaurar imagem original\n"
              << " 18 - Salvar imagem atual em JPEG\n"
              << " 0  - Sair\n"
              << "================================================\n"
              << "Opcao: ";
}

static int readInt() {
    int v;
    while (!(std::cin >> v)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Valor invalido, tente novamente: ";
    }
    return v;
}

static double readDouble() {
    double v;
    while (!(std::cin >> v)) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Valor invalido, tente novamente: ";
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <imagem_entrada>" << std::endl;
        return 1;
    }

    g_original = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (g_original.empty()) {
        std::cerr << "Erro: nao foi possivel abrir a imagem '" << argv[1] << "'." << std::endl;
        return 1;
    }
    g_current = g_original.clone();

    cv::namedWindow(WIN_ORIGINAL, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(WIN_RESULT, cv::WINDOW_AUTOSIZE);
    cv::imshow(WIN_ORIGINAL, g_original);
    refresh();

    bool running = true;
    while (running) {
        printMenu();
        int opt = readInt();

        switch (opt) {
            case 1: { // brilho
                std::cout << "Delta de brilho [-255,255]: ";
                int delta = readInt();
                g_current = pointops::adjustBrightness(g_current, delta);
                refresh();
                break;
            }
            case 2: { // contraste
                std::cout << "Fator de contraste (0,255]: ";
                double factor = readDouble();
                g_current = pointops::adjustContrast(g_current, factor);
                refresh();
                break;
            }
            case 3: { // negativo
                g_current = pointops::negative(g_current);
                refresh();
                break;
            }
            case 4: { // mostrar histograma
                cv::Mat lum = imgops::toGrayscaleLuminance1C(g_current);
                histo::Hist256 h = histo::computeHistogram(lum);
                cv::Mat histImg = histo::drawHistogram(h);
                cv::namedWindow(WIN_HIST, cv::WINDOW_AUTOSIZE);
                cv::imshow(WIN_HIST, histImg);
                cv::waitKey(1);
                break;
            }
            case 5: { // equalizar (cinza ou cor via luminancia)
                g_current = histo::equalizeColorViaLuminance(g_current);
                refresh();
                break;
            }
            case 6: { // equalizar em Lab (extra)
                g_current = labops::equalizeLab(g_current);
                refresh();
                break;
            }
            case 7: { // histogram matching
                std::cout << "Caminho da imagem de referencia (tons de cinza): ";
                std::string refPath;
                std::cin >> refPath;
                cv::Mat refImg = cv::imread(refPath, cv::IMREAD_COLOR);
                if (refImg.empty()) {
                    std::cerr << "Nao foi possivel abrir '" << refPath << "'." << std::endl;
                    break;
                }
                cv::Mat srcGray = imgops::toGrayscaleLuminance1C(g_current);
                cv::Mat refGray = imgops::toGrayscaleLuminance1C(refImg);
                cv::Mat matched1C = histo::histogramMatching(srcGray, refGray);
                cv::Mat matched3C;
                cv::merge(std::vector<cv::Mat>{matched1C, matched1C, matched1C}, matched3C);
                g_current = matched3C;
                refresh();
                break;
            }
            case 8: { // converter para cinza
                g_current = imgops::toGrayscaleLuminance(g_current);
                refresh();
                break;
            }
            case 9: { // quantizar
                std::cout << "Numero maximo de tons (n): ";
                int n = readInt();
                cv::Mat gray = imgops::toGrayscaleLuminance(g_current); // idempotente se ja for cinza
                g_current = imgops::quantize(gray, n);
                refresh();
                break;
            }
            case 10: { // espelhar horizontal
                g_current = imgops::mirror(g_current, true, false);
                refresh();
                break;
            }
            case 11: { // espelhar vertical
                g_current = imgops::mirror(g_current, false, true);
                refresh();
                break;
            }
            case 12: { // zoom out
                std::cout << "Fator de reducao sx (>=1): ";
                double sx = readDouble();
                std::cout << "Fator de reducao sy (>=1): ";
                double sy = readDouble();
                if (sx < 1.0 || sy < 1.0) {
                    std::cerr << "sx e sy devem ser >= 1." << std::endl;
                    break;
                }
                g_current = geom::zoomOut(g_current, sx, sy);
                refresh();
                break;
            }
            case 13: { // zoom in 2x
                g_current = geom::zoomIn2x(g_current);
                refresh();
                break;
            }
            case 14: { // rotacionar horario
                g_current = geom::rotate90(g_current, true);
                refresh();
                break;
            }
            case 15: { // rotacionar anti-horario
                g_current = geom::rotate90(g_current, false);
                refresh();
                break;
            }
            case 16: { // convolucao
                std::vector<conv::Kernel3x3> kernels = conv::builtinKernels();
                std::cout << "Kernels disponiveis:\n";
                for (size_t k = 0; k < kernels.size(); ++k) {
                    std::cout << "  " << (k + 1) << " - " << kernels[k].name << "\n";
                }
                std::cout << "  " << (kernels.size() + 1) << " - Kernel customizado (informar os 9 pesos)\n";
                std::cout << "Escolha: ";
                int kchoice = readInt();

                double w[3][3];
                bool addOffset = false;
                bool colorAllowed = false;

                if (kchoice >= 1 && static_cast<size_t>(kchoice) <= kernels.size()) {
                    const conv::Kernel3x3& k = kernels[kchoice - 1];
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            w[i][j] = k.w[i][j];
                    addOffset = k.addOffset127;
                    colorAllowed = k.colorAllowed;
                } else {
                    std::cout << "Informe os 9 pesos, linha a linha (a b c / d e f / g h i):\n";
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            w[i][j] = readDouble();
                    std::cout << "Somar 127 ao resultado antes do clamping? (0=nao, 1=sim): ";
                    addOffset = (readInt() != 0);
                    colorAllowed = false;
                }

                if (colorAllowed && g_current.channels() == 3) {
                    g_current = conv::convolve3x3Color(g_current, w, addOffset);
                } else {
                    cv::Mat gray1C = imgops::toGrayscaleLuminance1C(g_current);
                    cv::Mat result1C = conv::convolve3x3Gray(gray1C, w, addOffset);
                    cv::merge(std::vector<cv::Mat>{result1C, result1C, result1C}, g_current);
                }
                refresh();
                break;
            }
            case 17: { // restaurar original
                g_current = g_original.clone();
                refresh();
                break;
            }
            case 18: { // salvar
                std::cout << "Nome do arquivo de saida (ex: resultado.jpg): ";
                std::string outPath;
                std::cin >> outPath;
                std::vector<int> jpegParams = { cv::IMWRITE_JPEG_QUALITY, 95 };
                bool ok = cv::imwrite(outPath, g_current, jpegParams);
                std::cout << (ok ? "Imagem salva em " + outPath
                                  : std::string("Erro ao salvar a imagem."))
                          << std::endl;
                break;
            }
            case 0: {
                running = false;
                break;
            }
            default:
                std::cout << "Opcao invalida." << std::endl;
        }
    }

    return 0;
}
