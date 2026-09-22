# INF01046 — Trabalho 2: Transformações Lineares, Equalização e Matching de Histograma, Convolução e Filtragem no Domínio Espacial

Extensão do Trabalho 1 em C++, usando **OpenCV** apenas como biblioteca de
leitura/gravação de arquivos (`imread`/`imwrite`) e como toolkit de janela
(`highgui`: janelas, trackbars, teclado). Todas as operações de
processamento de imagem são implementadas manualmente.

A interface é **inteiramente dentro das janelas do OpenCV** — sem prompts
no console durante o uso. Parâmetros contínuos (brilho, contraste, níveis
de quantização, fatores de zoom, escolha de kernel) são trackbars; as
demais operações são atalhos de teclado, lidos num loop que chama
`cv::waitKey` continuamente. Isso é importante: uma versão anterior lia
esses parâmetros via `std::cin`, o que bloqueava a thread principal e
fazia o gerenciador de janelas mostrar "not responding" enquanto o
programa esperava você digitar algo no terminal — esse problema não existe
mais nesta versão.

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
    └── main.cpp                      # trackbars + atalhos de teclado, tudo no OpenCV
```

## Como compilar

```bash
cd trabalho2_pdi
make
```
Gera o executável em `build/trabalho2`.

Pré-requisitos: `g++` com C++17 e `libopencv-dev` (`sudo apt install libopencv-dev g++` no Ubuntu/Debian).

## Como executar

```bash
./build/trabalho2 caminho/para/imagem.jpg
# ou, para poder usar o histogram matching ('m'):
./build/trabalho2 caminho/para/imagem.jpg caminho/para/referencia.jpg
```

Se aparecer erro de "symbol lookup error" envolvendo `/snap/...`, o
problema é o `LD_LIBRARY_PATH` do seu terminal (normalmente por ter sido
aberto a partir de um app instalado via snap), não o programa — rode
`LD_LIBRARY_PATH= ./build/trabalho2 imagem.jpg` para confirmar, ou abra um
terminal "limpo" do sistema.

O programa abre três janelas:
- **Original** — a imagem carregada, fixa.
- **Resultado** — pré-visualização ao vivo (clique nela para que o teclado funcione) e o resultado acumulado das operações.
- **Controles** — as trackbars.

Uma janela **Histograma** aparece à parte quando você aperta `h`.

### Trackbars (janela "Controles")
| Trackbar | Efeito |
|---|---|
| Brilho | ajuste em tempo real, -255..255 |
| Contraste | ajuste em tempo real, fator 0.01..5.00 |
| Niveis | níveis de quantização, usado ao apertar `u` |
| ZoomSx / ZoomSy | fatores de redução, usados ao apertar `z` |
| Kernel | 0=nenhum, 1..7=kernel de convolução (ver legenda abaixo), usado ao apertar `k` |

Legenda dos kernels: `1=Gaussiano(passa-baixas) 2=Laplaciano 3=PassaAltasGenérico 4=PrewittHx 5=PrewittHy 6=SobelHx 7=SobelHy`.

### Atalhos de teclado (com a janela "Resultado" em foco)
| Tecla | Ação |
|---|---|
| `n` | negativo |
| `g` | converter para tons de cinza (luminância) |
| `e` | equalizar histograma (cinza / cor via luminância) |
| `l` | equalizar histograma em L\*a\*b\* **[pontos extra]** |
| `u` | quantizar tons (usa a trackbar "Niveis") |
| `x` | espelhar horizontal |
| `y` | espelhar vertical |
| `z` | zoom out / reduzir (usa "ZoomSx"/"ZoomSy") |
| `Z` (shift+z) | zoom in 2x2 |
| `r` | rotacionar 90° horário |
| `R` (shift+r) | rotacionar 90° anti-horário |
| `k` | aplicar convolução com o kernel da trackbar |
| `m` | histogram matching com a imagem de referência (2º argumento da linha de comando) |
| `h` | mostrar histograma em janela separada |
| `o` | restaurar imagem original |
| `s` | salvar imagem atual (gera `resultado_NNN.jpg`, numeração automática) |
| `?` | reimprime esta lista de comandos no console |
| `q` / `ESC` | sair |

Cada operação de tecla parte do estado atual (você pode compor: por
exemplo, `e` para equalizar, depois `r` para rotacionar, depois `k` com um
kernel de Sobel selecionado). O ajuste de brilho/contraste pelas trackbars
é incorporado automaticamente à imagem de trabalho antes de qualquer outra
operação, então nada se perde.

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
exaustiva 256×256, perfeitamente rápida para este propósito).

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
original vs. resultado (histogramas antes/depois da equalização, exemplos
de cada kernel de convolução, zoom in/out, rotações, etc.); uma captura da
interface (janelas "Original" + "Resultado" + "Controles"); dificuldades
enfrentadas e o que você faria diferente.