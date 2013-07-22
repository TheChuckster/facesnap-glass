#include <stdio.h>
 
const int N = 16;
 
__global__ 
void hello(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}
 
int cuda_test()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
	char* ad = NULL;
	int* bd = NULL;
 
	printf("%s", a);
 
	cudaMalloc((void**)&ad, sizeof(a)); 
	cudaMalloc((void**)&bd, sizeof(b)); 

	cudaMemcpy(ad, a, sizeof(a), cudaMemcpyHostToDevice); 
	cudaMemcpy(bd, b, sizeof(b), cudaMemcpyHostToDevice); 
	
	dim3 dimBlock(N, 1);
	dim3 dimGrid(1, 1);
	hello<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy(a, ad, sizeof(a), cudaMemcpyDeviceToHost); 

	cudaFree(ad);
	cudaFree(bd);
	
	printf("%s\n", a);
	
	return EXIT_SUCCESS;
}
