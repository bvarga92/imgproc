/***************************************************************************************************************/
/* Hozza kell adni a projekthez a DevIL konyvtarakat. Pl. x64 eseten:                                          */
/*   - C/C++ --> General --> Additional Include Directories: .\DevIL_x64\include\                              */
/*   - Linker --> General --> Additional Library Directories: .\DevIL_x64                                      */
/*   - Linker --> Input --> Additional Dependencies: DevIL.lib;ILU.lib;ILUT.lib;                               */
/* A .dll fajlokat be kell masolni oda, ahova a leforditott .exe kerul (megfelelo Debug vagy Release mappa).   */
/* Project --> Properties --> C/C++ --> Language menuben be kell kapcsolni az OpenMP tamogatast!               */
/* Csak szines, 24 bites szinmelysegu kepeket hasznaljunk!                                                     */
/* Idomeres eredmenyei:                                                                                        */
/*    - sima: 9.5 Mpixel/s                                                                                     */
/*    - OMP: 19.8 Mpixel/s                                                                                     */
/*    - SSE: 18.1 Mpixel/s                                                                                     */
/*    - SSE+OMP: 44.4 Mpixel/s                                                                                 */
/***************************************************************************************************************/

#include "memory.h"
#include "time.h"
#include "omp.h"
#include <IL/ilut.h>
#include <IL/ilu.h>
#include "emmintrin.h"
#include "nmmintrin.h"
#include "sobel.h"

#define RUNS 5 //futtatasok szama a pontosabb idomereshez
void (*func)(int,int,const float*,float*)=&sobel_sse_omp; //melyik implementaciot hasznaljuk
#define INPUTFILE   "_images/img1.bmp" //bemeneti fajlnev
#define OUTPUTFILE  "_images/img1_sobel.bmp" //kimeneti fajlnev

int main(){
	/* DevIL inicializalasa */
	ilInit();
	iluInit();
	ILuint ilImg=0;
	ilGenImages(1,&ilImg);
	ilBindImage(ilImg);
	if(!ilLoadImage((const wchar_t*)INPUTFILE)){
		printf("Cannot open input file.\n");
		return 1;
	}
	/* kep parameterei */
	ILubyte* imgData=ilGetData(); 
	int imgWidth=ilGetInteger(IL_IMAGE_WIDTH);
	int imgHeight=ilGetInteger(IL_IMAGE_HEIGHT);
	ILint imgOrigin=ilGetInteger(IL_ORIGIN_MODE);
	int imgWidthF=imgWidth+FILTER_W-1;
	int imgHeightF=imgHeight+FILTER_H-1;
	int imgFOffsetW=(FILTER_W-1)/2;
	int imgFOffsetH=(FILTER_H-1)/2;
	printf("Input resolution: %4dx%4d\n",imgWidth,imgHeight);
	/* tomb az uj kepnek, kinullazzuk */
	float *imgFloat=(float*)(_aligned_malloc(4*imgWidthF*imgHeightF*sizeof(float),32));
	for(int row=0; row<imgHeightF; row++)
		for(int col=0; col<imgWidthF;col++){
			int pixel=(row*imgWidthF+col)*4;
			imgFloat[pixel+0]=0.0f;
			imgFloat[pixel+1]=0.0f;
			imgFloat[pixel+2]=0.0f;
			imgFloat[pixel+3]=0.0f;
		}
	/* atmasoljuk a kepet a kiterjesztett tombbe */
	for(int row=0; row<imgHeight; row++)
		for(int col=0; col<imgWidth; col++){
			int pixel_dst=((row+imgFOffsetH)*imgWidthF+(col+imgFOffsetW))*4;
			int pixel_src=(row*imgWidth+col)*3;
			imgFloat[pixel_dst+0]=(float)imgData[pixel_src+0]; //r
			imgFloat[pixel_dst+1]=(float)imgData[pixel_src+1]; //g
			imgFloat[pixel_dst+2]=(float)imgData[pixel_src+2]; //b
			imgFloat[pixel_dst+3]=0.0;                         //alpha
		}
	/* kepfeldolgozas es idomeres */
	clock_t s0, e0;
	double d0, mpixel;
	float *imgFloatRes=(float*)_aligned_malloc(4*imgWidthF*imgHeightF*sizeof(float),32);
	s0=clock();
	for(int r=0; r<RUNS; r++) func(imgHeight, imgWidth, imgFloat, imgFloatRes);
	e0=clock();
	d0=(double)(e0-s0)/(RUNS*CLOCKS_PER_SEC);
	mpixel=(imgWidth*imgHeight/d0)/1000000;
	printf("CPU time: %4.4f\n",d0);
	printf("Mpixel/s: %4.4f\n",mpixel);
	/* eredmeny kiirasa */
	for(int row=0; row<imgHeight; row++)
		for(int col=0; col<imgWidth; col++){
			int pixel_dst=(row*imgWidth+col)*3;
			int pixel_src=((row+imgFOffsetH)*imgWidthF+(col+imgFOffsetW))*4;
			imgData[pixel_dst+0]=(ILubyte)imgFloatRes[pixel_src+0];
			imgData[pixel_dst+1]=(ILubyte)imgFloatRes[pixel_src+1];
			imgData[pixel_dst+2]=(ILubyte)imgFloatRes[pixel_src+2];
		}
	_aligned_free(imgFloat);
	_aligned_free(imgFloatRes);
	ilSetData(imgData);
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage((const wchar_t*)OUTPUTFILE);
	ilDeleteImages(1,&ilImg);
	return 0;
}
