#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <helper_timer.h>
#include "colour-convert.h"

// number of threads
// change it for experiment
int T = 1000;

// load empty kernal
__global__ void mykernel(void)
{
    
}



__global__ void rgb2yuvKernel(unsigned char *imgr,unsigned char *imgg,unsigned char *imgb,unsigned char *imgy,unsigned char *imgcb,unsigned char *imgcr) {

    unsigned char r, g, b;
    unsigned char y, cb, cr;

    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
   

    r = imgr[index];
    g = imgg[index];
    b = imgb[index];
    
    y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
    cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
    cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
    
    imgy[index] = y;
    imgcb[index] = cb;
    imgcr[index] = cr;
    

}
 
 
__global__ void yuv2rgbKernel(unsigned char *imgr,unsigned char *imgg,unsigned char *imgb,unsigned char *imgy,unsigned char *imgcb,unsigned char *imgcr) {

    int  rt,gt,bt;
    int y, cb, cr;    
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;


    y  = (int)imgy[index];
    cb = (int)imgcb[index] - 128;
    cr = (int)imgcr[index] - 128;
    
    rt  = (int)( y + 1.402*cr);
    if(rt > 255)
        rt = 255;
    if(rt < 0)
        rt =  0;
    gt  = (int)( y - 0.344*cb - 0.714*cr); 
    if(gt > 255)
        gt =  255;
    if(gt < 0)
        gt =  0;           
    bt  = (int)( y + 1.772*cb);
    if(bt > 255)
        bt =  255;
    if(bt < 0)
        bt = 0;


    imgr[index] = rt;
    imgg[index] = gt;
    imgb[index] = bt;
}




PPM_IMG copy_and_return_PPM(PPM_IMG img_in)
{
    PPM_IMG img_out;
    StopWatchInterface *timer=NULL;

    int TOTAL_PIXEL = img_in.w*img_in.h;

    int size = TOTAL_PIXEL * sizeof(char);


    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(size);
    img_out.img_g = (unsigned char *)malloc(size);
    img_out.img_b = (unsigned char *)malloc(size);

    //Put you CUDA initialization code here.
    unsigned char *r_d, *g_d, *b_d;
    unsigned char *rr_d, *gg_d, *bb_d;
    
    cudaMalloc((void **)&r_d, size);
    cudaMalloc((void **)&g_d, size);
    cudaMalloc((void **)&b_d, size);

    cudaMalloc((void **)&rr_d, size);
    cudaMalloc((void **)&gg_d, size);
    cudaMalloc((void **)&bb_d, size);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Copy RGB inputs from host to device
    cudaMemcpy(r_d, img_in.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d, img_in.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, img_in.img_b, size, cudaMemcpyHostToDevice);

    sdkStopTimer(&timer);

    printf("Copy time(Host to Device): %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

   
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Copy YUV output from device to host
    cudaMemcpy(img_out.img_r, r_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, g_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, b_d, size, cudaMemcpyDeviceToHost);

    sdkStopTimer(&timer);
    printf("Copy time(Device to Host): %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // freeing intermediate memory
    cudaFree(r_d);cudaFree(g_d);cudaFree(b_d);cudaFree(rr_d);cudaFree(gg_d);cudaFree(bb_d);

    return img_out;    
}




// we call this function to load empty kernel 
void load_empty_kernel()
{
     mykernel<<<1,1>>>();
}



YUV_IMG rgb2yuvGPU(PPM_IMG img_in)
{

    YUV_IMG img_out;
    //Put you CUDA initialization code here.
    
    unsigned char *d_r, *d_g, *d_b;
    unsigned char *d_y, *d_cb, *d_cr;

    img_out.w = img_in.w;
    img_out.h = img_in.h;


    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);


    cudaMalloc((void **)&d_r, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc((void **)&d_g, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc((void **)&d_b, sizeof(unsigned char)*img_out.w*img_out.h);

    
    cudaMalloc((void **)&d_y, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc((void **)&d_cb, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc((void **)&d_cr, sizeof(unsigned char)*img_out.w*img_out.h);   


    cudaMemcpy(d_r, img_in.img_r, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, img_in.img_g, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, img_in.img_b, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    

    rgb2yuvKernel<<<(img_in.w*img_in.h)/T,T>>>(d_r,d_g,d_b,d_y,d_cb,d_cr);//Launch the Kernel


    cudaMemcpy(img_out.img_y, d_y, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, d_cb, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, d_cr, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);

    cudaFree(d_r);cudaFree(d_g);cudaFree(d_b);cudaFree(d_y);cudaFree(d_cb);cudaFree(d_cr);
    return img_out;
}




PPM_IMG yuv2rgbGPU(YUV_IMG img_in)
{
    PPM_IMG img_out;
    //Put you CUDA setup code here.

    unsigned char *d_r, *d_g, *d_b;
    unsigned char *d_y, *d_cb, *d_cr;

    img_out.w = img_in.w;
    img_out.h = img_in.h;


    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);


    cudaMalloc((void **)&d_r, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc((void **)&d_g, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc((void **)&d_b, sizeof(unsigned char)*img_out.w*img_out.h);

    
    cudaMalloc((void **)&d_y, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc((void **)&d_cb, sizeof(unsigned char)*img_out.w*img_out.h);
    cudaMalloc((void **)&d_cr, sizeof(unsigned char)*img_out.w*img_out.h);   


    cudaMemcpy(d_y, img_in.img_y, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cb, img_in.img_u, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cr, img_in.img_v, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyHostToDevice);
    

    yuv2rgbKernel<<<(img_in.w*img_in.h)/T,T>>>(d_r,d_g,d_b,d_y,d_cb,d_cr);//Launch the Kernel


    cudaMemcpy(img_out.img_r, d_r, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, d_g, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, d_b, sizeof(unsigned char)*img_out.w*img_out.h, cudaMemcpyDeviceToHost);
   
    
    
    cudaFree(d_r);cudaFree(d_g);cudaFree(d_b);cudaFree(d_y);cudaFree(d_cb);cudaFree(d_cr);
    return img_out;
}



//Convert RGB to YUV444, all components in [0, 255]
YUV_IMG rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
 
    for(i = 0; i < img_out.w*img_out.h; i ++){
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        img_out.img_y[i] = y;
        img_out.img_u[i] = cb;
        img_out.img_v[i] = cr;
    }
    
    return img_out;
}



unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}



//Convert YUV to RGB, all components in [0, 255]
PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int i;
    int  rt,gt,bt;
    int y, cb, cr;
    
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr); 
        bt  = (int)( y + 1.772*cb);

        img_out.img_r[i] = clip_rgb(rt);
        img_out.img_g[i] = clip_rgb(gt);
        img_out.img_b[i] = clip_rgb(bt);
    }
    
    return img_out;
}
