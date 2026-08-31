# Definições de variáveis
CXX = g++
CXXFLAGS = -Wall -std=c++17
TARGET = programa
SRC = main.cpp

# Regra padrão executada ao digitar apenas 'make'
all: $(TARGET) run

# Compila o executável
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Executa o programa compilado
run: $(TARGET)
	./$(TARGET)

# Limpa o arquivo executável gerado
clean:
	rm -f $(TARGET)
