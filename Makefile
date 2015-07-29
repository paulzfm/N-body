nbody: util.o run.o nbody.o
	nvcc util.o run.o nbody.o -o nbody -lpthread -lX11 -arch=sm_30

util.o: util.c util.h
	gcc -c util.c -o util.o

run.o: run.cu run.h
	nvcc -c run.cu -o run.o

nbody.o: nbody.c
	gcc -c nbody.c -o nbody.o

clean:
	rm -rf nbody *~ *.o
