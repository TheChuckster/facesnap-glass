#include <assert.h>
#include <helper_cuda.h>

#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void PowKernel(float * Ximg,float * Yimg ,float *d_Dst,int imageW,int imageH)
{
 int Row = blockIdx.y*blockDim.y+threadIdx.y;
 int Col = blockIdx.x*blockDim.x+threadIdx.x;
 d_Dst[Row*imageW+Col] = Ximg[Row*imageW+Col]*Yimg[Row*imageW+Col];
}

extern "C" void PowGPU(float * Ximg,float * Yimg,float *d_Dist,int imageW,int imageH)
{
 dim3 blocks(imageW / ROWS_BLOCKDIM_X, imageH / ROWS_BLOCKDIM_Y);
 dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
 PowKernel<<<blocks, threads>>>(Ximg,Yimg,d_Dist,imageW,imageH);
 getLastCudaError("convolutionRowsKernel() execution failed\n");
}

