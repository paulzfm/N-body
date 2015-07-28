nbody: nbody.cu util.c util.h
	nvcc nbody.cu util.c -o nbody -arch=sm_30

clean:
	rm -rf nbody *~
