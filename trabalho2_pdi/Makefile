# ============================================================================
# INF01046 - Trabalho 2: Transformacoes Lineares, Equalizacao e Matching de
# Histograma, Convolucao e Filtragem no Dominio Espacial
#
# Uso:
#   make          -> compila o executavel "trabalho2"
#   make clean    -> remove os arquivos gerados
#
# Requisitos: g++ com C++17 e OpenCV (pkg-config opencv4 ou opencv).
# ============================================================================

CXX := g++
CXXFLAGS := -std=c++17 -Wall -O2 -Isrc

OPENCV_PKG := $(shell pkg-config --exists opencv4 && echo opencv4 || echo opencv)
CXXFLAGS += $(shell pkg-config --cflags $(OPENCV_PKG))
LDFLAGS  := $(shell pkg-config --libs $(OPENCV_PKG))

BUILD_DIR := build
SRC_DIR := src

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

TARGET := $(BUILD_DIR)/trabalho2

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
