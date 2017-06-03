#include "omp.h"
#include "emmintrin.h"
#include "nmmintrin.h"
#include "conv_filter.h"

const float filter_laplace[25]={-1.0, -1.0, -1.0, -1.0, -1.0,
                                -1.0, -1.0, -1.0, -1.0, -1.0,
                                -1.0, -1.0, 24.0, -1.0, -1.0,
                                -1.0, -1.0, -1.0, -1.0, -1.0,
                                -1.0, -1.0, -1.0, -1.0, -1.0};

const float filter_sobel_x[25]={0.020833, 0.041666, 0.000000, -0.041666, -0.020833,
                                0.083333, 0.166666, 0.000000, -0.166666, -0.083333,
                                0.125000, 0.250000, 0.000000, -0.250000, -0.125000,
                                0.083333, 0.166666, 0.000000, -0.166666, -0.083333,
                                0.020833, 0.041666, 0.000000, -0.041666, -0.020833};

const float filter_sobel_y[25]={-0.020833, -0.083333, -0.125000, -0.083333, -0.020833,
                                -0.041666, -0.166666, -0.250000, -0.166666, -0.041666,
                                 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
                                 0.041666,  0.166666,  0.250000,  0.166666,  0.041666,
                                 0.020833,  0.083333,  0.125000,  0.083333,  0.020833};

const float filter_horiz_edges[25]={0.0, 0.0, -1.0, 0.0, 0.0,
                                    0.0, 0.0, -1.0, 0.0, 0.0,
                                    0.0, 0.0,  2.0, 0.0, 0.0,
                                    0.0, 0.0,  0.0, 0.0, 0.0,
                                    0.0, 0.0,  0.0, 0.0, 0.0};

const float filter_vert_edges[25]={ 0.0,  0.0, 0.0, 0.0, 0.0,
                                    0.0,  0.0, 0.0, 0.0, 0.0,
                                   -1.0, -1.0, 2.0, 0.0, 0.0,
                                    0.0,  0.0, 0.0, 0.0, 0.0,
                                    0.0,  0.0, 0.0, 0.0, 0.0};

const float filter_average[25]={0.04, 0.04, 0.04, 0.04, 0.04,
                                0.04, 0.04, 0.04, 0.04, 0.04,
                                0.04, 0.04, 0.04, 0.04, 0.04,
                                0.04, 0.04, 0.04, 0.04, 0.04,
                                0.04, 0.04, 0.04, 0.04, 0.04};

const float filter_gauss_lpf[25]={0.003663, 0.014652, 0.025641, 0.014652, 0.003663,
                                  0.014652, 0.058608, 0.095238, 0.058608, 0.014652,
                                  0.025641, 0.095238, 0.150183, 0.095238, 0.025641,
                                  0.014652, 0.058608, 0.095238, 0.058608, 0.014652,
                                  0.003663, 0.014652, 0.025641, 0.014652, 0.003663};

const float filter_highpass[25]={-0.04, -0.04, -0.04, -0.04, -0.04,
                                 -0.04, -0.04, -0.04, -0.04, -0.04,
                                 -0.04, -0.04,  1.96, -0.04, -0.04,
                                 -0.04, -0.04, -0.04, -0.04, -0.04,
                                 -0.04, -0.04, -0.04, -0.04, -0.04};

const float filter_motionblur[25]={0.2, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.2, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.2, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.2, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.2};

const float filter_emboss[25]={0.0,  0.0,  0.0, 0.0, 0.0,
                               0.0, -2.0, -1.0, 0.0, 0.0,
                               0.0, -1.0,  1.0, 1.0, 0.0,
                               0.0,  0.0,  1.0, 2.0, 0.0,
                               0.0,  0.0,  0.0, 0.0, 0.0};

const float filter_emboss2[25]={-1.0, -1.0, -1.0, -1.0, .0,
                                -1.0, -1.0, -1.0,  0.0, 1.0,
                                -1.0, -1.0,  0.0,  1.0, 1.0,
                                -1.0,  0.0,  1.0,  1.0, 1.0,
                                 0.0,  1.0,  1.0,  1.0, 1.0};

void conv_filter(int imgHeight, int imgWidth, const float *filter, const float *imgFloatSrc, float *imgFloatDst){
	int imgWidthF=imgWidth+FILTER_W-1; //kiterjesztett kep szelessege
	int imgHeightF=imgHeight+FILTER_H-1; //kiterjesztett kep magassaga
	int imgFOffsetW=(FILTER_W-1)/2; //oldalso kiterjesztesek nagysaga
	int imgFOffsetH=(FILTER_H-1)/2; //also-felso kiterjesztesek nagysaga
	for(int row=imgFOffsetH; row<(imgHeight+imgFOffsetH); row++){
		for(int col=imgFOffsetW; col<(imgWidth+imgFOffsetW); col++){
			float fval[4]={0.0f, 0.0f, 0.0f, 0.0f}; //akkumulator
			/* konvolucio mindharom komponensre */
			for(int fy=0; fy<FILTER_H; fy++){
				int y=row+fy-imgFOffsetH;
				for(int fx=0; fx<FILTER_W; fx++){
					int x=col+fx-imgFOffsetW;
					fval[0]+=(filter[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 0]; //r
					fval[1]+=(filter[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 1]; //g
					fval[2]+=(filter[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 2]; //b
				}
			}
			/* kimeneti pixel irasa (0...255 tartomanyba csonkolni) */
			for(int i=0; i<4; i++){
				if(fval[i]>255.0)
					imgFloatDst[(row*imgWidthF+col)*4 + i]=255.0;
				else if(fval[i]<0.0)
					imgFloatDst[(row*imgWidthF+col)*4 + i]=0.0;
				else
					imgFloatDst[(row*imgWidthF+col)*4 + i]=fval[i];
			}
		}
	}
}

void conv_filter_omp(int imgHeight, int imgWidth, const float *filter, const float *imgFloatSrc, float *imgFloatDst){
	int imgWidthF=imgWidth+FILTER_W-1; //kiterjesztett kep szelessege
	int imgHeightF=imgHeight+FILTER_H-1; //kiterjesztett kep magassaga
	int imgFOffsetW=(FILTER_W-1)/2; //oldalso kiterjesztesek nagysaga
	int imgFOffsetH=(FILTER_H-1)/2; //also-felso kiterjesztesek nagysaga
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic,1) nowait
		for(int row=imgFOffsetH; row<(imgHeight+imgFOffsetH); row++){
			for(int col=imgFOffsetW; col<(imgWidth+imgFOffsetW); col++){
				float fval[4]={0.0f, 0.0f, 0.0f, 0.0f}; //akkumulator
				/* konvolucio mindharom komponensre */
				for(int fy=0; fy<FILTER_H; fy++){
					int y=row+fy-imgFOffsetH;
					for(int fx=0; fx<FILTER_W; fx++){
						int x=col+fx-imgFOffsetW;
						fval[0]+=(filter[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 0]; //r
						fval[1]+=(filter[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 1]; //g
						fval[2]+=(filter[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 2]; //b
					}
				}
				/* kimeneti pixel irasa (0...255 tartomanyba csonkolni) */
				for(int i=0; i<4; i++){
					if(fval[i]>255.0)
						imgFloatDst[(row*imgWidthF+col)*4 + i]=255.0;
					else if(fval[i]<0.0)
						imgFloatDst[(row*imgWidthF+col)*4 + i]=0.0;
					else
						imgFloatDst[(row*imgWidthF+col)*4 + i]=fval[i];
				}
			}
		}
	}
}

void conv_filter_sse(int imgHeight, int imgWidth, const float *filter, const float *imgFloatSrc, float *imgFloatDst){
	int imgWidthF=imgWidth+FILTER_W-1; //kiterjesztett kep szelessege
	int imgHeightF=imgHeight+FILTER_H-1; //kiterjesztett kep magassaga
	int imgFOffsetW=(FILTER_W-1)/2; //oldalso kiterjesztesek nagysaga
	int imgFOffsetH=(FILTER_H-1)/2; //also-felso kiterjesztesek nagysaga
	/* konstansok a csonkolashoz */
	__declspec(align(16)) __m128 const_0=_mm_set_ps(0.0, 0.0, 0.0, 0.0);
	__declspec(align(16)) __m128 const_255=_mm_set_ps(255.0, 255.0, 255.0, 255.0);
	/* szuroegyutthatok negyszerezese */
	float filter_ext[4*FILTER_W*FILTER_H];
	for(int i=0; i<4*FILTER_W*FILTER_H; i++) filter_ext[i]=filter[i/4];
	/* float --> m128 konverzio */
	__declspec(align(16)) __m128 filter_l[FILTER_W*FILTER_H];
	for(int i=0; i<4*FILTER_W*FILTER_H; i++) filter_l[i>>2].m128_f32[i&0x3]=filter_ext[i];
	/* kepfeldolgozas */
	for(int row=imgFOffsetH; row<(imgHeight+imgFOffsetH); row++){
		for(int col=imgFOffsetW; col<(imgWidth+imgFOffsetW); col++){
			__declspec(align(16)) __m128 fval=const_0; //akkumulator
			/* konvolucio mindharom komponensre */
			for(int fy=0; fy<FILTER_H; fy++){
				int y=row+fy-imgFOffsetH;
				for(int fx=0; fx<FILTER_W; fx++){
					int x=col+fx-imgFOffsetW;
					__m128 pixel=_mm_load_ps(imgFloatSrc+(y*imgWidthF+x)*4);
					fval=_mm_add_ps(fval,_mm_mul_ps(filter_l[fy*FILTER_W+fx],pixel));
				}
			}
			/* kimeneti pixel irasa (0...255 tartomanyba csonkolni) */
			fval=_mm_min_ps(fval,const_255);
			fval=_mm_max_ps(fval,const_0);
			_mm_stream_ps(imgFloatDst+(row*imgWidthF+col)*4,fval); //store is lehetne, de igy kikeruli a cache-t
		}
	}
}

void conv_filter_sse_omp(int imgHeight, int imgWidth, const float *filter, const float *imgFloatSrc, float *imgFloatDst){
	int imgWidthF=imgWidth+FILTER_W-1; //kiterjesztett kep szelessege
	int imgHeightF=imgHeight+FILTER_H-1; //kiterjesztett kep magassaga
	int imgFOffsetW=(FILTER_W-1)/2; //oldalso kiterjesztesek nagysaga
	int imgFOffsetH=(FILTER_H-1)/2; //also-felso kiterjesztesek nagysaga
	/* konstansok a csonkolashoz */
	__declspec(align(16)) __m128 const_0=_mm_set_ps(0.0, 0.0, 0.0, 0.0);
	__declspec(align(16)) __m128 const_255=_mm_set_ps(255.0, 255.0, 255.0, 255.0);
	/* szuroegyutthatok negyszerezese */
	float filter_ext[4*FILTER_W*FILTER_H];
	for(int i=0; i<4*FILTER_W*FILTER_H; i++) filter_ext[i]=filter[i/4];
	/* float --> m128 konverzio */
	__declspec(align(16)) __m128 filter_l[FILTER_W*FILTER_H];
	for(int i=0; i<4*FILTER_W*FILTER_H; i++) filter_l[i>>2].m128_f32[i&0x3]=filter_ext[i];
	/* kepfeldolgozas */
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic,1) nowait
		for(int row=imgFOffsetH; row<(imgHeight+imgFOffsetH); row++){
			for(int col=imgFOffsetW; col<(imgWidth+imgFOffsetW); col++){
				__declspec(align(16)) __m128 fval=const_0; //akkumulator
				/* konvolucio mindharom komponensre */
				for(int fy=0; fy<FILTER_H; fy++){
					int y=row+fy-imgFOffsetH;
					for(int fx=0; fx<FILTER_W; fx++){
						int x=col+fx-imgFOffsetW;
						__m128 pixel=_mm_load_ps(imgFloatSrc+(y*imgWidthF+x)*4);
						fval=_mm_add_ps(fval,_mm_mul_ps(filter_l[fy*FILTER_W+fx],pixel));
					}
				}
				/* kimeneti pixel irasa (0...255 tartomanyba csonkolni) */
				fval=_mm_min_ps(fval,const_255);
				fval=_mm_max_ps(fval,const_0);
				_mm_stream_ps(imgFloatDst+(row*imgWidthF+col)*4,fval); //store is lehetne, de igy kikeruli a cache-t
			}
		}
	}
}
