nbody: util.o run.o nbody.o
	nvcc util.o run.o nbody.o -o nbody -lpthread -lX11 -arch=sm_30 -O2

util.o: util.cpp util.h
	nvcc -c util.cpp -o util.o -O2

run.o: run.cu run.h
	nvcc -c run.cu -o run.o -O2

nbody.o: nbody.cu
	nvcc -c nbody.cu -o nbody.o -O2

clean:
	rm -rf nbody *~ *.o
