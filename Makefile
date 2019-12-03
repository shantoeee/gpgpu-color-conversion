# Build tools
CFLAGS= -I/usr/local/cuda-10.0/samples/common/inc
NVCC = /usr/local/cuda-10.0/bin/nvcc


# here are all the objects
GPUOBJS = main.o colour-convert.o
OBJS = cuda.o

# make and compile
mycode.out: $(GPUOBJS)
	$(NVCC) -o mycode $(GPUOBJS) 

main.o: main.cpp
	$(NVCC) -c main.cpp $(CFLAGS)

colour-convert.o: colour-convert.cu
	$(NVCC) -c colour-convert.cu $(CFLAGS)

clean:
	rm -f out_rgb.ppm out_rgb_cpu.ppm out_yuv_cpu.yuv out_rgb.ppm out_yuv.yuv out_rgb_gpu.ppm out_rgb_gpu_copy_test.ppm out_yuv_gpu.yuv mycode $(GPUOBJS)

