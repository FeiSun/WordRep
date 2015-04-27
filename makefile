CXX = g++
CXXFLAGS = -I/usr/local/include/eigen/ -std=c++11 -O3 -march=native -funroll-loops 
LDFLAGS = -lm -fopenmp

all: w2v

w2v : main.cpp WordRep.cpp
	$(CXX) main.cpp WordRep.cpp -o w2v $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -rf w2v
