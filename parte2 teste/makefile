CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 $(shell pkg-config --cflags opencv4)
LDLIBS = $(shell pkg-config --libs opencv4)

# Nome do executavel e do arquivo fonte
TARGET = Trabalho1
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)