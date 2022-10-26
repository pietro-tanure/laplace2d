compiler = mpic++
flags = -I. -I./eigen/

headers = $(wildcard *.hpp)
sources = $(wildcard *.cpp)
objects = $(sources:.cpp=.o)

executables: solver

%.o: %.cpp $(headers)
	$(compiler) -c -o $@ $< $(flags)

solver: $(objects)
	$(compiler) -o $@ $^ $(flags)

clean:
	rm -f solver *.o
