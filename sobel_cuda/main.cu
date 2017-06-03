/***************************************************************************************************************/
/* Visual Studio projekt beallitasok:                                                                          */
/*   - CUDA 8.0 Runtime projekt, x64 platform                                                                  */
/*   - CUDA C/C++ --> Additional Include Directories: $(ProjectDir)DevIL_x64\include                           */
/*   - Linker --> General --> Additional Library Directories: $(ProjectDir)DevIL_x64                           */
/*   - Linker --> Input --> Additional Dependencies: DevIL.lib;ILU.lib;ILUT.lib                                */
/*   - CUDA Linker beallitasoknal ugyanigy                                                                     */
/*   - VC++ Directories --> Include Directories: $(ProjectDir)DevIL_x64\include (csak az IntelliSense miatt)   */
/* A .dll fajlokat be kell masolni oda, ahova a leforditott .exe kerul (megfelelo Debug vagy Release mappa).   */
/* Csak szines, 24 bites szinmelysegu kepeket hasznaljunk!                                                     */
/* Idomeres eredmenye: 211.7 Mpixel/s                                                                          */
/***************************************************************************************************************/

#include "time.h"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <IL/ilut.h>
#include <IL/ilu.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define RUNS 10
#define FILTER_W 5
#define FILTER_H 5
#define INPUTFILE   "_images/img1.jpg"
#define OUTPUTFILE  "_images/img1_sobel.jpg"
#define KERN_32X8 1

const float filter_sobel_x[25]={0.041666f, 0.083332f, 0.000000f, -0.083332f, -0.041666f,
                                0.166666f, 0.333332f, 0.000000f, -0.333332f, -0.166666f,
                                0.250000f, 0.500000f, 0.000000f, -0.500000f, -0.250000f,
                                0.166666f, 0.333332f, 0.000000f, -0.333332f, -0.166666f,
                                0.041666f, 0.083332f, 0.000000f, -0.083332f, -0.041666f};

const float filter_sobel_y[25]={-0.041666f, -0.166666f, -0.250000f, -0.166666f, -0.041666f,
                                -0.083332f, -0.333332f, -0.500000f, -0.333332f, -0.083332f,
                                 0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f,
                                 0.083332f,  0.333332f,  0.500000f,  0.333332f,  0.083332f,
                                 0.041666f,  0.166666f,  0.250000f,  0.166666f,  0.041666f};

extern __global__ void sobel_kernel_16x16(unsigned char* gInput, unsigned char* gOutput, float *coeffsX, float *coeffsY, int imgWidthF);
extern __global__ void sobel_kernel_32x8(unsigned char* gInput, unsigned char* gOutput, float *coeffsX, float *coeffsY, int imgWidthF);

void convCUDA(int imgHeight, int imgWidth, unsigned char *imgSrc, unsigned char *imgDst){
	int imgWidthF=imgWidth+FILTER_W-1;
	int imgHeightF=imgHeight+FILTER_H-1;
	unsigned char *device_imgSrc, *device_imgDst;
	float *device_coeffsX, *device_coeffsY;
	/* ha tobb GPU van, akkor az elsot valasztjuk */
	cudaSetDevice(0);
	/* kernel parameterek */
	cudaMalloc((void**)&device_imgSrc,imgWidthF*imgHeightF*3);
	cudaMalloc((void**)&device_imgDst,imgWidthF*imgHeightF*3);
	cudaMalloc((void**)&device_coeffsX,FILTER_W*FILTER_H*sizeof(float));
	cudaMalloc((void**)&device_coeffsY,FILTER_W*FILTER_H*sizeof(float));
	cudaMemcpy(device_imgSrc,imgSrc,imgWidthF*imgHeightF*3,cudaMemcpyHostToDevice);
	cudaMemcpy(device_coeffsX,filter_sobel_x,FILTER_W*FILTER_H*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(device_coeffsY,filter_sobel_y,FILTER_W*FILTER_H*sizeof(float),cudaMemcpyHostToDevice);
	/* ha az L1 cache es a megosztott memoria kozos, akkor a nagyobb megosztott memoriat preferaljuk */
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	/* kernel futtatasa es idomeres */
	clock_t s0, e0;
	double d0;
	s0=clock();
	#if KERN_32X8
		dim3 thrBlock(32,8);
		dim3 thrGrid(imgWidth/32,imgHeight/8);
		for(int i=0;i<RUNS;i++) sobel_kernel_32x8<<<thrGrid,thrBlock>>>(device_imgSrc,device_imgDst,device_coeffsX,device_coeffsY,imgWidthF);
	#else
		dim3 thrBlock(16,16);
		dim3 thrGrid(imgWidth/16,imgHeight/16);
		for(int i=0;i<RUNS;i++) sobel_kernel_16x16<<<thrGrid,thrBlock>>>(device_imgSrc,device_imgDst,device_coeffsX,device_coeffsY,imgWidthF);
	#endif
	cudaThreadSynchronize();
	e0=clock();
	d0=(double)(e0-s0)/CLOCKS_PER_SEC;
	printf("Total kernel time: %6.0f ms (%d runs)\n",d0*1000,RUNS);
	printf("Mpixel/s: %4.4f\n",RUNS*(imgWidth*imgHeight/d0)/1000000);
	/* eredmeny masolasa, befejezes */
	cudaMemcpy(imgDst,device_imgDst,imgWidthF*imgHeightF*3,cudaMemcpyDeviceToHost);
	cudaFree(device_imgSrc);
	cudaFree(device_imgDst);
	cudaFree(device_coeffsX);
	cudaFree(device_coeffsY);
	cudaDeviceReset();
}

int main(){
	/* DevIL inicializalasa */
	ilInit();
	iluInit();
	ILuint ilImg = 0;
	ilGenImages(1, &ilImg);
	ilBindImage(ilImg);
	if (!ilLoadImage((const char*)INPUTFILE)){
		printf("Cannot open input file.\n");
		return 1;
	}
	/* kep parameterei */
	ILubyte* imgData = ilGetData();
	int imgWidth = ilGetInteger(IL_IMAGE_WIDTH);
	int imgHeight = ilGetInteger(IL_IMAGE_HEIGHT);
	ILint imgOrigin = ilGetInteger(IL_ORIGIN_MODE);
	int imgWidthF = imgWidth + FILTER_W - 1;
	int imgHeightF = imgHeight + FILTER_H - 1;
	int imgFOffsetW = (FILTER_W - 1) / 2;
	int imgFOffsetH = (FILTER_H - 1) / 2;
	printf("Input resolution: %4dx%4d\n", imgWidth, imgHeight);
	/* tomb az uj kepnek, kinullazzuk */
	unsigned char *imgSrc = (unsigned char*)(_aligned_malloc(3 * imgWidthF*imgHeightF*sizeof(unsigned char), 32));
	for (int row = 0; row<imgHeightF; row++)
		for (int col = 0; col<imgWidthF; col++){
			int pixel = (row*imgWidthF + col) * 3;
			imgSrc[pixel + 0] = 0.0f;
			imgSrc[pixel + 1] = 0.0f;
			imgSrc[pixel + 2] = 0.0f;
		}
	/* atmasoljuk a kepet a kiterjesztett tombbe */
	for (int row = 0; row<imgHeight; row++)
		for (int col = 0; col<imgWidth; col++){
			int pixel_dst = ((row + imgFOffsetH)*imgWidthF + (col + imgFOffsetW)) * 3;
			int pixel_src = (row*imgWidth + col) * 3;
			imgSrc[pixel_dst + 0] = (unsigned char)imgData[pixel_src + 0]; //r
			imgSrc[pixel_dst + 1] = (unsigned char)imgData[pixel_src + 1]; //g
			imgSrc[pixel_dst + 2] = (unsigned char)imgData[pixel_src + 2]; //b
		}
	/* kepfeldolgozas */
	unsigned char *imgRes = (unsigned char*)_aligned_malloc(4 * imgWidthF*imgHeightF*sizeof(unsigned char), 32);
	convCUDA(imgHeight, imgWidth, imgSrc, imgRes);
	/* eredmeny kiirasa */
	for (int row = 0; row<imgHeight; row++)
		for (int col = 0; col<imgWidth; col++){
			int pixel_dst = (row*imgWidth + col) * 3;
			int pixel_src = ((row + imgFOffsetH)*imgWidthF + (col + imgFOffsetW)) * 3;
			imgData[pixel_dst + 0] = (ILubyte)imgRes[pixel_src + 0];
			imgData[pixel_dst + 1] = (ILubyte)imgRes[pixel_src + 1];
			imgData[pixel_dst + 2] = (ILubyte)imgRes[pixel_src + 2];
		}
	_aligned_free(imgSrc);
	_aligned_free(imgRes);
	ilSetData(imgData);
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage((const char*)OUTPUTFILE);
	ilDeleteImages(1, &ilImg);
	return 0;
}
