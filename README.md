GPU Programming – Colour Space Conversion
Tutorial: https://sharifuli.com/2018/02/23/gpgpu-cuda-color-space/

- This program converts image from RGV to YUV and vice versa. 
- There are two types of implementations, on CPU and on GPU
- There are functionalities for time recording to record time for CPU and GPU operation


To run:
-------

1. Command for compilation: "make"

2. Command for running the program (after compilation): "./mycode"

3. Command for cleaning the previous output (should be run initially): "make clean"

4. Sample Output and Execution Steps:

--Cleaning the previous output

askapoor@asb10928u-c04:~/Downloads/886-a2$ make clean
rm -f out_rgb.ppm out_rgb_cpu.ppm out_yuv_cpu.yuv out_rgb.ppm out_yuv.yuv out_rgb_gpu.ppm out_rgb_gpu_copy_test.ppm out_yuv_gpu.yuv mycode main.o colour-convert.o

--Compiling the Code

abc@def:~/Downloads/886-a2$ make
/usr/local/cuda-9.1/bin/nvcc -c main.cpp -I/usr/local/cuda-9.1/samples/common/inc
/usr/local/cuda-9.1/bin/nvcc -c colour-convert.cu -I/usr/local/cuda-9.1/samples/common/inc
/usr/local/cuda-9.1/bin/nvcc -o mycode main.o colour-convert.o 

--Running the Code

abc@def:~/Downloads/886-a2$ ./mycode
Running colour space converter .
Image size: 1000 x 700

--Copy Output

Starting GPU copy testing...
Copy time(Host to Device): 0.232000 (ms)
Copy time(Device to Host): 0.402000 (ms)

--RGB to YUV (In GPU)

Starting GPU processing...
RGB to YUV conversion time(GPU): 1.838000 (ms)

--Comparing with similar CPU output

Comparing the output with CPU Output
2219 pixels are not same for rgb to yuv 
0.317000 percentage of pixels are not same for rgb to yuv 
1 Max Difference for Pixel Y
1 Max Difference for Pixel U
0 Max Difference for Pixel V
Not similar by small percentage

--YUV to RGB (In GPU)

YUV to RGB conversion time(GPU): 1.914000 (ms)

--Comparing with similar CPU output

Comparing the output with CPU Output
2219 pixels are not same for yuv to rgb 
0.317000 percentage of pixels are not same for yuv to rgb
1 Max Difference for Pixel R
1 Max Difference for Pixel G
2 Max Difference for Pixel B
Not similar by small percentage

--CPU Processing time for both the operation

Starting CPU processing...
RGB to YUV conversion time: 19.236000 (ms)
YUV to RGB conversion time: 11.514000 (ms)


5. Out of Memory Issues (Blacked-Out Image): When testing with the larger image please make sure that there is enough memory in the system and GPU. Otherwise the output would be Blacked-Out from certain parts and a chance of Segmentation Fault.
