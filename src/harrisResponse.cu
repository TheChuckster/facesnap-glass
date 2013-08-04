#include <assert.h>
#include <helper_cuda.h>

#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void harrisResponseKernel(float * Ximg,float * Yimg ,float * XYimg,float*C,float *H,int imageW,int imageH)
{
 int Row = blockIdx.y*blockDim.y+threadIdx.y;
 int Col = blockIdx.x*blockDim.x+threadIdx.x;
 int k=Row*imageW+Col;
 int max=50000;
 C[Row*imageW+Col]=0;
 // H=(IxG.*IyG - IxyG.^2) - k*(IxG + IyG).^2;
 H[Row*imageW+Col] = Ximg[Row*imageW+Col]*Yimg[Row*imageW+Col]-XYimg[Row*imageW+Col]*XYimg[Row*imageW+Col] -0.04*(Ximg[Row*imageW+Col]+Yimg[Row*imageW+Col])*(Ximg[Row*imageW+Col]+Yimg[Row*imageW+Col]);
 if(H[k]>max)
 C[k]=H[k];
}

extern "C" void harrisResponse(float * Ximg,float * Yimg,float *XYimg,float * C,float *H,int imageW,int imageH)
{
 dim3 blocks(imageW / ROWS_BLOCKDIM_X, imageH / ROWS_BLOCKDIM_Y);
 dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
 harrisResponseKernel<<<blocks, threads>>>(Ximg,Yimg,XYimg,C,H,imageW,imageH);
 getLastCudaError("convolutionRowsKernel() execution failed\n");
}
