nbody: nbody.o util.o
	nvcc nbody.o util.o -o nbody -lpthread -lX11

nbody.o: nbody.cu util.h
	nvcc -c nbody.cu -o nbody.o

util.o: util.c util.h
	nvcc -c util.c -o util.o -lX11

clean:
	rm -rf nbody *~ *.o
