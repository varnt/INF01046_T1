#include <iostream>
#include <fstream>
#include <string>
#include <opencv4/opencv.h>


using namespace std;
using namespace cv;

int main() {
    std::cout << "Oiêeee" << std::endl;
    string filenameInput = "Space_187k.jpg";
    string filenameOutput = "Space_187k_output.jpg";
    


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