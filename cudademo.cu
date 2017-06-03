#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_ELEMENTS 100 //vektorok elemszama

/* CUDA kernel */
__global__ void vecadd(const int* a, const int* b, int* c){
	int thId=threadIdx.x;
	c[thId]=a[thId]+b[thId];
}

/* foprogram */
int main(){
	int a[NUM_ELEMENTS], b[NUM_ELEMENTS], c[NUM_ELEMENTS];
	int *bufferA, *bufferB, *bufferC;
	int i;
	/* ha tobb GPU van, akkor az elsot valasztjuk */
	cudaSetDevice(0);
	/* kernel parameterek atadasa */
	srand(123);
	for(i=0;i<NUM_ELEMENTS;i++){
		a[i]=rand()%1000;
		b[i]=rand()%1000;
	}
	cudaMalloc((void**)&bufferA,sizeof(int)*NUM_ELEMENTS);
	cudaMalloc((void**)&bufferB,sizeof(int)*NUM_ELEMENTS);
	cudaMalloc((void**)&bufferC,sizeof(int)*NUM_ELEMENTS);
	cudaMemcpy(bufferA,a,sizeof(int)*NUM_ELEMENTS,cudaMemcpyHostToDevice);
	cudaMemcpy(bufferB,b,sizeof(int)*NUM_ELEMENTS,cudaMemcpyHostToDevice);
	/* kernel futtatasa NUM_ELEMENTS darab szallal */
	vecadd<<<1,NUM_ELEMENTS>>>(bufferA,bufferB,bufferC);
	if (cudaGetLastError() != cudaSuccess) {
		printf("Error: failed to launch kernel.\n");
		cudaFree(bufferA);
		cudaFree(bufferB);
		cudaFree(bufferC);
		return 1;
	}
	cudaDeviceSynchronize();
	/* eredmeny kiolvasasa */
	cudaMemcpy(c,bufferC,sizeof(int)*NUM_ELEMENTS,cudaMemcpyDeviceToHost);
	/* befejezes */
	cudaFree(bufferA);
	cudaFree(bufferB);
	cudaFree(bufferC);
	printf("\n");
	for(i=0;i<NUM_ELEMENTS;i++) printf("%3d + %3d = %4d\n",a[i],b[i],c[i]);
	getchar();
	cudaDeviceReset();
	return 0;
}
