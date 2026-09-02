#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "ImageOps.h"

using namespace std;
using namespace cv;

int main(){
/*int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <imagem_entrada>" << std::endl;
        return 1;
    }
 
    string filenameInput = argv[1];
    */

    cout << "1º Trabalho de Implementação - INF01046 – Fundamentos de Processamento de Imagens - Aluno: Victor de Souza Arnt Matrícula:00291097" << std::endl;
    string filenameInput = "Space_187k.jpg"; 
    string filenameOutput;

    
    size_t lastDot = filenameInput.find_last_of(".");

    if (lastDot != std::string::npos) {
       //arquivo com extensão, adiciona "_out" antes da extensão
        filenameOutput = filenameInput.substr(0, lastDot) + "_out" + filenameInput.substr(lastDot);
    } else {
        //Caso arquivo n tenha estensão, adiciona "_out"
        filenameOutput = filenameInput + "_out";
    }


    //Abrir o arquivo de entrada para leitura
    ifstream entrada(filenameInput);
    if (!entrada.is_open()) {
        cout << "Erro ao abrir o arquivo de entrada: " << filenameInput << endl;
        return 1;
    }

    //Cria novo arquivo para escrita
    ofstream saida(filenameOutput);
    if (!saida.is_open()) {
        cout << "Erro ao abrir o arquivo de saída: " << filenameOutput << endl;
        entrada.close();
        return 1;
    }
    string linha;
    while (getline(entrada, linha)) {
        saida << linha << endl;
    }
    entrada.close();
    saida.close();

    cout << "Aqruivo entrada copiado em arquivo saida com  sucesso!" << endl;
    return 0;
}