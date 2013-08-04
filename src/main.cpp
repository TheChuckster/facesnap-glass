// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "convolutionSeparable_common.h"
#include "CImg.h"

using namespace cimg_library;

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolutionColumnCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void PowGPU(
        float * Ximg,
        float * Yimg, 
        float  *d_Dist,
        int imageW,
        int imageH
);

extern "C" void harrisResponse(
       float * Ximg,
       float * Yimg,
       float * XYimg,
       float * C,
       float * H,
       int imageW,
       int imageH
);




int main( int argc, char** argv) 
{
    float *h_KernelX,*h_DataA,*h_DataB,*Xh_ResultGPU,*Xhp_ResultGPU,*Yh_ResultGPU;
    float *YY_ResultGPU,*XX_ResultGPU,*XY_ResultGPU,*h_KernelY,*g_Kernel;
    float *XXg_ResultGPU,*YYg_ResultGPU,*XYg_ResultGPU,*C,*H;
    printf("[%s] - Starting...\n", argv[0]);
    CImg<float> image("lena.jpg"), visu(500,400,1,3,0);
    int width = image.width(); 
    int height = image.height();
    int size=width*height;
    printf("Initialization...\n");
    
    h_DataA       =  (float *)malloc(size* sizeof(float));
    h_DataB       =  (float *)malloc(size* sizeof(float));
    Xh_ResultGPU  =  (float *)malloc(size* sizeof(float));
    Yh_ResultGPU  =  (float *)malloc(size* sizeof(float));
    YY_ResultGPU  =  (float *)malloc(size* sizeof(float));
    XX_ResultGPU  =  (float *)malloc(size* sizeof(float));
    XY_ResultGPU  =  (float *)malloc(size* sizeof(float));
    XXg_ResultGPU =  (float *)malloc(size* sizeof(float));
    YYg_ResultGPU =  (float *)malloc(size* sizeof(float));
    XYg_ResultGPU =  (float *)malloc(size* sizeof(float));
    C             =  (float *)malloc(size* sizeof(float));
    H             =  (float *)malloc(size* sizeof(float));
    
    h_DataA = image.data();
    
    //X edge detection 3x3
    
    printf("X edge detection 3x3\n");
    h_KernelX = (float *)malloc(9*sizeof(float));
    h_KernelX[0]=-1;h_KernelX[1]=0;h_KernelX[2]=1;
    h_KernelX[3]=-1; h_KernelX[4]=0;h_KernelX[5]=1;
    h_KernelX[6]=-1;h_KernelX[7]=0;h_KernelX[8]=1;
    convolutionRowCPU(h_DataB,h_DataA,h_KernelX,width,height,3);
    convolutionColumnCPU(Xh_ResultGPU,h_DataB,h_KernelX,width,height,3);
    
    //Y edge detection 3x3
    
    printf("Y edge detection 3x3\n");
    h_KernelY = (float *)malloc(9*sizeof(float));
    h_KernelY[0]=-1;h_KernelY[1]=-1;h_KernelY[2]=-1;
    h_KernelY[3]=0; h_KernelY[4]=0;h_KernelY[5]=0;
    h_KernelY[6]=1;h_KernelY[7]=1;h_KernelY[8]=1;
    convolutionRowCPU(h_DataB,h_DataA,h_KernelY,width,height,3);
    convolutionColumnCPU(Yh_ResultGPU,h_DataB,h_KernelY,width,height,3);
    PowGPU(Xh_ResultGPU,Xh_ResultGPU,XX_ResultGPU,width,height);
    PowGPU(Yh_ResultGPU,Yh_ResultGPU,XX_ResultGPU,width,height);
    PowGPU(Yh_ResultGPU,Yh_ResultGPU,XX_ResultGPU,width,height);
    
    // Gausian Filter 
    
    printf("Gausian Filter 3x3\n");
    g_Kernel = (float *)malloc(9*sizeof(float));
    g_Kernel[0]=0.0585498;g_Kernel[1]=0.0965324;g_Kernel[2]=0.0585498;
    g_Kernel[3]=0.0965324; g_Kernel[4]=0.159155;g_Kernel[5]=0.0965324;
    g_Kernel[6]=0.0585498;g_Kernel[7]=0.0965324;g_Kernel[8]=0.0585498;
    convolutionRowCPU(h_DataB,h_DataA,g_Kernel,width,height,3);
    convolutionColumnCPU(XXg_ResultGPU,h_DataB,g_Kernel,width,height,3);
    convolutionRowCPU(h_DataB,h_DataA,g_Kernel,width,height,3);
    convolutionColumnCPU(YYg_ResultGPU,h_DataB,g_Kernel,width,height,3);
    convolutionRowCPU(h_DataB,h_DataA,g_Kernel,width,height,3);
    convolutionColumnCPU(XYg_ResultGPU,h_DataB,g_Kernel,width,height,3);
    
    // Harris Response
      
    printf("Harris Response\n");
    harrisResponse(XXg_ResultGPU, YYg_ResultGPU,XYg_ResultGPU,C,H,width,height);
    printf("Stop.\n");
     
     
    CImgDisplay main_disp(image,"Click a point");
    const unsigned char red[] = { 255,0,0 }, green[] = { 0,255,0 }, blue[] = { 0,0,255 };
    while (!main_disp.is_closed()) {
    main_disp.wait();
    }

    cudaFree(h_DataA);
    cudaFree(h_DataB);
    cudaFree(Xh_ResultGPU);
    cudaFree(Yh_ResultGPU);
    cudaFree(YY_ResultGPU);
    cudaFree(XX_ResultGPU);
    cudaFree(XY_ResultGPU);
    cudaFree(YYg_ResultGPU);
    cudaFree(XXg_ResultGPU); 
    cudaFree(XYg_ResultGPU); 
    cudaFree(C);
    cudaFree(H);

 return 0;
}
