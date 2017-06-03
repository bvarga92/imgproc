#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <IL/ilut.h>
#include <IL/ilu.h>

#define IMAGE_W   1280
#define IMAGE_H    720
#define THRESHOLD 3600
#define INPUTFILE  "_images/img1.jpg"
#define OUTPUTFILE "_images/img1_sobel.jpg"

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

float sobelRT(float r, float g, float b){
	static float buffer[5][IMAGE_W];
	static unsigned row=0, col=0;
	static bool wait=true;
	/* 5 sornyi buffert toltunk a vilagossagadatokkal */
	buffer[row][col]=0.2989*r+0.5870*g+0.1140*b;
	col++;
	if(col==IMAGE_W){
		col=0;
		row=(row+1)%5;
		if(row==0) wait=false;
	}
	if(wait) return 0;
	/* konvolucio */
	float accX=0, accY=0;
	for(int i=0;i<5;i++)
		for(int j=0;j<5;j++){
			accX+=filter_sobel_x[j*5+i]*buffer[(row+j)%5][col+i];
			accY+=filter_sobel_y[j*5+i]*buffer[(row+j)%5][col+i];
		}
	/* a ket eredmeny kombinalasa es kuszobozes */
	return ((accX*accX+accY*accY)>THRESHOLD)?255:0;
}

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
	ILint imgOrigin=ilGetInteger(IL_ORIGIN_MODE);
	if((ilGetInteger(IL_IMAGE_WIDTH)!=IMAGE_W)||(ilGetInteger(IL_IMAGE_HEIGHT)!=IMAGE_H)){
		printf("Error: resolution must be 1280x720.\n");
		ilDeleteImages(1,&ilImg);
		return 1;
	}
	/* tomb az uj kepnek */
	unsigned char *imgSrc=(unsigned char*)(_aligned_malloc(3*IMAGE_W*IMAGE_H*sizeof(unsigned char),32));
	for(int row=0; row<IMAGE_H; row++)
		for(int col=0; col<IMAGE_W;col++){
			int pixel=(row*IMAGE_W+col)*3;
			imgSrc[pixel+0]=(unsigned char)imgData[pixel+0]; //r
			imgSrc[pixel+1]=(unsigned char)imgData[pixel+1]; //g
			imgSrc[pixel+2]=(unsigned char)imgData[pixel+2]; //b
		}
	/* kepfeldolgozas (kesleltetes kompenzalva, szelek levagva) */
	unsigned char *imgRes=(unsigned char*)_aligned_malloc(3*IMAGE_W*IMAGE_H*sizeof(unsigned char),32);
	for(int row=0;row<IMAGE_H;row++)
		for(int col=0;col<IMAGE_W;col++){
			int rd=(row*IMAGE_W+col)*3;
			int wr=((row-2)*IMAGE_W+col+2)*3;
			if(wr<0) wr=0;
			imgRes[wr]=imgRes[wr+1]=imgRes[wr+2]=(unsigned char)sobelRT(imgSrc[rd],imgSrc[rd+1],imgSrc[rd+2])*(col<IMAGE_W-5);
		}
	/* eredmeny kiirasa */
	for(int row=0; row<IMAGE_H; row++)
		for(int col=0; col<IMAGE_W; col++){
			int pixel=(row*IMAGE_W+col)*3;
			imgData[pixel+0]=(ILubyte)imgRes[pixel+0];
			imgData[pixel+1]=(ILubyte)imgRes[pixel+1];
			imgData[pixel+2]=(ILubyte)imgRes[pixel+2];
		}
	_aligned_free(imgSrc);
	_aligned_free(imgRes);
	ilSetData(imgData);
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage((const wchar_t*)OUTPUTFILE);
	ilDeleteImages(1,&ilImg);
	return 0;
}
