nbody: util.o run.o nbody.o
	nvcc util.o run.o nbody.o -o nbody -lpthread -lX11 -arch=sm_30

util.o: util.cpp util.h
	nvcc -c util.cpp -o util.o

run.o: run.cu run.h
	nvcc -c run.cu -o run.o

nbody.o: nbody.cpp
	nvcc -c nbody.cpp -o nbody.o

clean:
	rm -rf nbody *~ *.o
