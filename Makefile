nbody: nbody.cu util.c util.h
	nvcc nbody.cu util.c -o nbody -arch=sm_30 -lpthread -lX11

clean:
	rm -rf nbody *~
