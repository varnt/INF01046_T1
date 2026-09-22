# INF01046 — Trabalho 2: Transformações Lineares, Equalização e Matching de Histograma, Convolução e Filtragem no Domínio Espacial

Extensão do Trabalho 1 em C++, usando **OpenCV** apenas como biblioteca de
leitura/gravação de arquivos (`imread`/`imwrite`) e como toolkit de janela
(`highgui`: janelas e `imshow`). Todas as operações de processamento de
imagem são implementadas manualmente.

## Estrutura dos arquivos

```
trabalho2_pdi/
├── Makefile
└── src/
    ├── ImageOps.h / .cpp       # (T1) espelhamento, luminância, quantização
    ├── PointOps.h / .cpp        # (1.2, 1.3, 1.4) brilho, contraste, negativo
    ├── Histogram.h / .cpp        # (1.1, 1.5, 1.6) histograma, equalização, matching
    ├── LabOps.h / .cpp            # (1.5 extra) equalização em L*a*b*
    ├── Geometry.h / .cpp           # (2.7, 2.8, 2.9) zoom out/in, rotação 90°
    ├── Convolution.h / .cpp         # (2.10) convolução 3x3 + kernels do enunciado
    └── main.cpp                      # menu no console + janelas do OpenCV
```

## Como compilar

```bash
cd trabalho2_pdi
make
```
Gera o executável em `build/trabalho2`.

Pré-requisitos: `g++` com C++17 e `libopencv-dev` instalado (`sudo apt install libopencv-dev cmake g++` no Ubuntu/Debian — o CMake não é necessário para este Makefile, mas o pacote de desenvolvimento do OpenCV sim).

## Como executar

```bash
./build/trabalho2 caminho/para/imagem.jpg
```

O programa abre duas janelas (**Original** e **Resultado**) e apresenta um
**menu numerado no console**. Cada opção aplica uma operação sobre a
imagem "de trabalho" (`g_current`), que começa como uma cópia da original
e vai sendo transformada a cada escolha — permitindo compor operações
(ex.: brilho, depois negativo, depois rotação) ou usar a opção **17** para
restaurar a imagem original a qualquer momento.

Principais opções do menu:
- **1/2/3** — brilho, contraste, negativo (Parte 1, itens 2–4)
- **4** — mostra o histograma da luminância da imagem atual em janela separada (item 1)
- **5** — equalização de histograma; funciona tanto para imagens em cinza quanto coloridas (usa o histograma cumulativo da luminância aplicado a cada canal) (item 5)
- **6** — equalização no espaço L\*a\*b\* (pontos extra)
- **7** — histogram matching com uma segunda imagem informada pelo caminho (item 6)
- **8/9** — conversão para cinza e quantização (reaproveitados do T1)
- **10/11** — espelhamento horizontal/vertical (reaproveitados do T1)
- **12/13** — zoom out (com sx, sy) e zoom in 2x2 (itens 7–8)
- **14/15** — rotação 90° horário/anti-horário, aplicável repetidamente (item 9)
- **16** — convolução 3x3, com os 7 kernels do enunciado prontos para escolher, ou pesos customizados (item 10)
- **18** — salva a imagem atual em JPEG

## Notas de implementação (para o relatório)

**Histograma (1)** — `Histogram::computeHistogram` conta a ocorrência de
cada um dos 256 tons de cinza; `drawHistogram` desenha as barras
normalizando pela altura máxima (`hist[i]/max * alturaÚtil`), sem usar
nenhuma função de desenho pronta do OpenCV (os pixels da imagem do
histograma são escritos diretamente).

**Brilho/contraste/negativo (2,3,4)** — operações de ponto aplicadas
canal a canal (B, G, R) de forma independente, com *clamping* explícito
para `[0,255]`.

**Equalização (5)** — implementa o histograma cumulativo clássico
`mapa[v] = round((cdf[v]-cdf_min)/(N-cdf_min) * 255)`. Para imagens
coloridas, o mapeamento é calculado a partir do histograma da luminância e
aplicado igualmente aos três canais, conforme pedido. A equalização em
L\*a\*b\* (pontos extra) converte manualmente BGR→XYZ→Lab (fórmulas
padrão CIE, D65), equaliza apenas o canal L e reconverte para BGR.

**Histogram matching (6)** — para cada tom da imagem de origem, procura o
tom da referência cujo histograma cumulativo mais se aproxima (busca
exaustiva 256×256, o que é perfeitamente rápido para este propósito).

**Zoom out (7)** — para cada pixel de saída, define um retângulo
`[round(i·sy), round((i+1)·sy))` × `[round(j·sx), round((j+1)·sx))` sobre a
imagem original (arredondado e recortado nos limites da imagem) e calcula
a média dos pixels cobertos, canal a canal.

**Zoom in 2x2 (8)** — feito em três passos: (1) posiciona os pixels
originais nas posições pares; (2) interpola horizontalmente as colunas
ímpares das linhas pares; (3) interpola verticalmente as linhas ímpares
usando as linhas pares já completas. A última linha/coluna, sem par
seguinte para interpolar, repete o último valor válido.

**Rotação 90° (9)** — troca linha por coluna: no sentido horário,
`dst(j, rows-1-i) = src(i,j)`; no anti-horário, `dst(cols-1-j, i) =
src(i,j)`. Chamar a função novamente sobre o resultado permite compor
rotações de 180°, 270° etc.

**Convolução 3x3 (10)** — segue exatamente a fórmula do enunciado
(`Conv(E) = i·A + h·B + g·C + f·D + e·E + d·F + c·G + b·H + a·I`), que
equivale a rotacionar o kernel 180° antes de aplicá-lo. As bordas da
imagem (1 pixel) são mantidas iguais à imagem de entrada, e a convolução é
aplicada apenas ao interior. Para os kernels (i)-(iii) o resultado é
diretamente limitado a `[0,255]`; para os kernels de gradiente (iv)-(vii)
soma-se 127 antes do *clamping*, como pedido. Apenas o kernel (i)
(Gaussiano) pode ser aplicado diretamente sobre a imagem colorida; os
demais são aplicados sobre a luminância.

## Sobre o relatório

Lembre de incluir, para cada item do trabalho: se foi concluído
satisfatoriamente (e, se não, por quê); capturas de tela mostrando
original vs. resultado (histogramas antes/depois da equalização,
exemplos de cada kernel de convolução, zoom in/out, rotações, etc.); uma
captura da interface (janelas "Original" + "Resultado" + menu do
console); dificuldades enfrentadas e o que você faria diferente.
