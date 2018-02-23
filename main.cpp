#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>
// helper for shared that are common to CUDA Samples
#include <helper_functions.h>
#include <helper_timer.h>

#include "colour-convert.h"

void run_copy_test(PPM_IMG img_in);
void run_cpu_color_test(PPM_IMG img_in);
void run_gpu_color_test(PPM_IMG img_in);


int main(){
    PPM_IMG img_ibuf_c;

    // load empty kernel
    load_empty_kernel();
    printf("Running colour space converter .\n");
    img_ibuf_c = read_ppm("in.ppm");

    // runs functions for testing copy time
    run_copy_test(img_ibuf_c);

    run_gpu_color_test(img_ibuf_c);
    run_cpu_color_test(img_ibuf_c);

    free_ppm(img_ibuf_c);
    
    return 0;
}


// we call this function to check 
// COPY time from host to device and vice versa
void run_copy_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_rgb_copy_test;
  
    printf("Starting GPU copy testing...\n");
    
    img_obuf_rgb_copy_test = copy_and_return_PPM(img_in); 

    write_ppm(img_obuf_rgb_copy_test, "out_rgb_gpu_copy_test.ppm");
    free_ppm(img_obuf_rgb_copy_test);
}


void run_gpu_color_test(PPM_IMG img_in)
{
    StopWatchInterface *timer=NULL;
    PPM_IMG img_obuf_rgb;
    YUV_IMG img_obuf_yuv;
  
    printf("\nStarting GPU processing...\n");
    

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    img_obuf_yuv = rgb2yuvGPU(img_in); //Start RGB 2 YUV
    sdkStopTimer(&timer);

    printf("\nRGB to YUV conversion time(GPU): %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);


    confirm_gpu_rgb2yuv(img_obuf_yuv,img_in);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
	
    img_obuf_rgb = yuv2rgbGPU(img_obuf_yuv); //Start YUV 2 RGB

    sdkStopTimer(&timer);

    printf("\nYUV to RGB conversion time(GPU): %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer); 

    confirm_gpu_yuv2rgb(img_obuf_rgb,img_in);
    



    write_yuv(img_obuf_yuv, "out_yuv_gpu.yuv");
    write_ppm(img_obuf_rgb, "out_rgb_gpu.ppm");
    
    free_ppm(img_obuf_rgb); //Uncomment these when the images exist
    free_yuv(img_obuf_yuv);

}


void run_cpu_color_test(PPM_IMG img_in)
{
    StopWatchInterface *timer=NULL;
    PPM_IMG img_obuf_rgb;
    YUV_IMG img_obuf_yuv;
    
    
    printf("\nStarting CPU processing...\n");
  
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    img_obuf_yuv = rgb2yuv(img_in); //Start RGB 2 YUV

    sdkStopTimer(&timer);
    printf("RGB to YUV conversion time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

   

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    img_obuf_rgb = yuv2rgb(img_obuf_yuv); //Start YUV 2 RGB

    sdkStopTimer(&timer);
    printf("YUV to RGB conversion time: %f (ms)\n\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);    

    write_yuv(img_obuf_yuv, "out_yuv.yuv");
    write_ppm(img_obuf_rgb, "out_rgb.ppm");
    
    free_ppm(img_obuf_rgb);
    free_yuv(img_obuf_yuv);
    
}



bool confirm_gpu_rgb2yuv(YUV_IMG gpu_img_in, PPM_IMG img_in) //Place code here that verifies your conversion
{

    YUV_IMG img_cpu_yuv;
    img_cpu_yuv = rgb2yuv(img_in);

    printf("comparing the output with CPU\n");

    int y_cpu, u_cpu, v_cpu, y_gpu, u_gpu, v_gpu;
    float j = 0;
    int max_diff_y = 0;
    int max_diff_u = 0;
    int max_diff_v = 0;
    for (int i = 0; i < gpu_img_in.h * gpu_img_in.w; i++)
    {

        y_cpu = (int)img_cpu_yuv.img_y[i];
        u_cpu = (int)img_cpu_yuv.img_u[i];
        v_cpu = (int)img_cpu_yuv.img_v[i];

        y_gpu = (int)gpu_img_in.img_y[i];
        u_gpu = (int)gpu_img_in.img_u[i];
        v_gpu = (int)gpu_img_in.img_v[i];
        //printf("%d\n",i);
        if((y_cpu != y_gpu) || (u_cpu != u_gpu) || (v_cpu != v_gpu))
        {

            if(max_diff_y < abs(y_cpu-y_gpu)){
                max_diff_y = (y_cpu-y_gpu);
            }
            if(max_diff_u < abs(u_cpu-u_gpu)){
                max_diff_u = (u_cpu-u_gpu);
            }
            if(max_diff_v < abs(v_cpu-v_gpu)){
                max_diff_v = (v_cpu-v_gpu);
            }
            j++;
        }
    }

    if (j > 0){
        printf("%d pixels are not same for rgb to yuv \n",(int)j);
        printf("%f percentage of pixels are not same for rgb to yuv \n", (j/(gpu_img_in.h * gpu_img_in.w))*100);
        printf("%d Max Difference for Pixel Y\n",max_diff_y);
        printf("%d Max Difference for Pixel U\n",max_diff_u);
        printf("%d Max Difference for Pixel V\n",max_diff_v);
        printf("Not similar by small percentage\n");
        return false;
    }
    printf("same\n");
    return true;
}



bool confirm_gpu_yuv2rgb(PPM_IMG gpu_img_in,PPM_IMG img_in) //Place code here that verifies your conversion
{

    YUV_IMG img_cpu_yuv;
    PPM_IMG img_cpu_rgb;
    img_cpu_yuv = rgb2yuv(img_in);
    img_cpu_rgb = yuv2rgb(img_cpu_yuv);
    printf("comparing the output with CPU\n");
  
    float j = 0;
    int r_cpu, g_cpu, b_cpu, r_gpu, g_gpu, b_gpu;
    int max_diff_r = 0;
    int max_diff_g = 0;
    int max_diff_b = 0;
    for (int i = 0; i < gpu_img_in.h * gpu_img_in.w; i++)
    {
        r_cpu = (int)img_cpu_rgb.img_r[i];
        g_cpu = (int)img_cpu_rgb.img_g[i];
        b_cpu = (int)img_cpu_rgb.img_b[i];

        r_gpu = (int)gpu_img_in.img_r[i];
        g_gpu = (int)gpu_img_in.img_g[i];
        b_gpu = (int)gpu_img_in.img_b[i];

        if((r_cpu != r_gpu) || (g_cpu != g_gpu) || (b_cpu != b_gpu))
        {

            if(max_diff_r < abs(r_cpu-r_gpu)){
                max_diff_r = (r_cpu-r_gpu);
            }
            if(max_diff_g < abs(g_cpu-g_gpu)){
                max_diff_g = (g_cpu-g_gpu);
            }
            if(max_diff_b < abs(b_cpu-b_gpu)){
                max_diff_b = (b_cpu-b_gpu);
            }
            j++;
        }
    }


    if (j > 0){
        printf("%d pixels are not same for yuv to rgb \n",(int)j);
        printf("%f percentage of pixels are not same for yuv to rgb\n", (j/(gpu_img_in.h * gpu_img_in.w))*100);
        printf("%d Max Difference for Pixel R\n",max_diff_r);
        printf("%d Max Difference for Pixel G\n",max_diff_g);
        printf("%d Max Difference for Pixel B\n",max_diff_b);
        printf("Not similar by small percentage\n");
        return false;
    }

    printf("same\n");
	return true;
}



PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);



    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}



void write_yuv(YUV_IMG img, const char * path){//Output in YUV444 Planar
    FILE * out_file;
    int i;
    

    out_file = fopen(path, "wb");
    fwrite(img.img_y,sizeof(unsigned char), img.w*img.h, out_file);
    fwrite(img.img_u,sizeof(unsigned char), img.w*img.h, out_file);
    fwrite(img.img_v,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}


void write_yuv2(YUV_IMG img, const char * path){ //Output in YUV444 Packed
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_y[i];
        obuf[3*i + 1] = img.img_u[i];
        obuf[3*i + 2] = img.img_v[i];
    }

    out_file = fopen(path, "wb");
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}


void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_yuv(YUV_IMG img)
{
    free(img.img_y);
    free(img.img_u);
    free(img.img_v);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}


