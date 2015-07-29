nbody: nbody.cu
	nvcc nbody.cu -o nbody -arch=sm_30 -lpthread -lX11

clean:
	rm -rf nbody *~ *.o
