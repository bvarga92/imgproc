#include "omp.h"
#include "emmintrin.h"
#include "nmmintrin.h"
#include <math.h>
#include "sobel.h"

const float filter_sobel_x[25]={0.041666, 0.083332, 0.000000, -0.083332, -0.041666,
                                0.166666, 0.333332, 0.000000, -0.333332, -0.166666,
                                0.250000, 0.500000, 0.000000, -0.500000, -0.250000,
                                0.166666, 0.333332, 0.000000, -0.333332, -0.166666,
                                0.041666, 0.083332, 0.000000, -0.083332, -0.041666};

const float filter_sobel_y[25]={-0.041666, -0.166666, -0.250000, -0.166666, -0.041666,
                                -0.083332, -0.333332, -0.500000, -0.333332, -0.083332,
                                 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,
                                 0.083332,  0.333332,  0.500000,  0.333332,  0.083332,
                                 0.041666,  0.166666,  0.250000,  0.166666,  0.041666};

void sobel(int imgHeight, int imgWidth, const float *imgFloatSrc, float *imgFloatDst){
	int imgWidthF=imgWidth+FILTER_W-1; //kiterjesztett kep szelessege
	int imgHeightF=imgHeight+FILTER_H-1; //kiterjesztett kep magassaga
	int imgFOffsetW=(FILTER_W-1)/2; //oldalso kiterjesztesek nagysaga
	int imgFOffsetH=(FILTER_H-1)/2; //also-felso kiterjesztesek nagysaga
	for(int row=imgFOffsetH; row<(imgHeight+imgFOffsetH); row++){
		for(int col=imgFOffsetW; col<(imgWidth+imgFOffsetW); col++){
			float fval_x[4]={0.0f, 0.0f, 0.0f, 0.0f};
			float fval_y[4]={0.0f, 0.0f, 0.0f, 0.0f};
			float fval[4]={0.0f, 0.0f, 0.0f, 0.0f};
			/* konvolucio mindharom komponensre */
			for(int fy=0; fy<FILTER_H; fy++){
				int y=row+fy-imgFOffsetH;
				for(int fx=0; fx<FILTER_W; fx++){
					int x=col+fx-imgFOffsetW;
					fval_x[0]+=(filter_sobel_x[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 0]; //r
					fval_x[1]+=(filter_sobel_x[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 1]; //g
					fval_x[2]+=(filter_sobel_x[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 2]; //b
					fval_y[0]+=(filter_sobel_y[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 0]; //r
					fval_y[1]+=(filter_sobel_y[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 1]; //g
					fval_y[2]+=(filter_sobel_y[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 2]; //b
				}
			}
			/* a ket eredmeny kombinalasa */
			fval[0]=sqrt(fval_x[0]*fval_x[0]+fval_y[0]*fval_y[0]); //r
			fval[1]=sqrt(fval_x[1]*fval_x[1]+fval_y[1]*fval_y[1]); //g
			fval[2]=sqrt(fval_x[2]*fval_x[2]+fval_y[2]*fval_y[2]); //b
			/* szurkearnyalatossa konvertalas */
			float gs=0.2989*fval[0]+0.5870*fval[1]+0.1140*fval[2];
			/* kuszobozes */
			gs=(gs>THRESHOLD)?(255.0):(0.0);
			/* kimeneti pixel irasa */
			for(int i=0; i<3; i++) imgFloatDst[(row*imgWidthF+col)*4 + i]=gs;
		}
	}
}

void sobel_omp(int imgHeight, int imgWidth, const float *imgFloatSrc, float *imgFloatDst){
	int imgWidthF=imgWidth+FILTER_W-1; //kiterjesztett kep szelessege
	int imgHeightF=imgHeight+FILTER_H-1; //kiterjesztett kep magassaga
	int imgFOffsetW=(FILTER_W-1)/2; //oldalso kiterjesztesek nagysaga
	int imgFOffsetH=(FILTER_H-1)/2; //also-felso kiterjesztesek nagysaga
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic,1) nowait
		for(int row=imgFOffsetH; row<(imgHeight+imgFOffsetH); row++){
			for(int col=imgFOffsetW; col<(imgWidth+imgFOffsetW); col++){
				float fval_x[4]={0.0f, 0.0f, 0.0f, 0.0f};
				float fval_y[4]={0.0f, 0.0f, 0.0f, 0.0f};
				float fval[4]={0.0f, 0.0f, 0.0f, 0.0f};
				/* konvolucio mindharom komponensre */
				for(int fy=0; fy<FILTER_H; fy++){
					int y=row+fy-imgFOffsetH;
					for(int fx=0; fx<FILTER_W; fx++){
						int x=col+fx-imgFOffsetW;
						fval_x[0]+=(filter_sobel_x[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 0]; //r
						fval_x[1]+=(filter_sobel_x[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 1]; //g
						fval_x[2]+=(filter_sobel_x[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 2]; //b
						fval_y[0]+=(filter_sobel_y[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 0]; //r
						fval_y[1]+=(filter_sobel_y[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 1]; //g
						fval_y[2]+=(filter_sobel_y[fy*FILTER_W+fx])*imgFloatSrc[(y*imgWidthF+x)*4 + 2]; //b
					}
				}
				/* a ket eredmeny kombinalasa */
				fval[0]=sqrt(fval_x[0]*fval_x[0]+fval_y[0]*fval_y[0]); //r
				fval[1]=sqrt(fval_x[1]*fval_x[1]+fval_y[1]*fval_y[1]); //g
				fval[2]=sqrt(fval_x[2]*fval_x[2]+fval_y[2]*fval_y[2]); //b
				/* szurkearnyalatossa konvertalas */
				float gs=0.2989*fval[0]+0.5870*fval[1]+0.1140*fval[2];
				/* kuszobozes */
				gs=(gs>THRESHOLD)?(255.0):(0.0);
				/* kimeneti pixel irasa */
				for(int i=0; i<3; i++) imgFloatDst[(row*imgWidthF+col)*4 + i]=gs;
			}
		}
	}
}

void sobel_sse(int imgHeight, int imgWidth, const float *imgFloatSrc, float *imgFloatDst){
	int imgWidthF=imgWidth+FILTER_W-1; //kiterjesztett kep szelessege
	int imgHeightF=imgHeight+FILTER_H-1; //kiterjesztett kep magassaga
	int imgFOffsetW=(FILTER_W-1)/2; //oldalso kiterjesztesek nagysaga
	int imgFOffsetH=(FILTER_H-1)/2; //also-felso kiterjesztesek nagysaga
	/* konstansok */
	__declspec(align(16)) __m128 const_0=_mm_set_ps(0.0, 0.0, 0.0, 0.0);
	__declspec(align(16)) __m128 const_255=_mm_set_ps(255.0, 255.0, 255.0, 255.0);
	__declspec(align(16)) __m128 const_rgb_to_gs=_mm_set_ps(0.0, 0.1140, 0.5870, 0.2989);
	__declspec(align(16)) __m128 thresh=_mm_set_ps(-1000, THRESHOLD, THRESHOLD, THRESHOLD);
	/* szuroegyutthatok negyszerezese */
	float sobel_x_ext[4*FILTER_W*FILTER_H], sobel_y_ext[4*FILTER_W*FILTER_H];
	for(int i=0; i<4*FILTER_W*FILTER_H; i++){
		sobel_x_ext[i]=filter_sobel_x[i>>2];
		sobel_y_ext[i]=filter_sobel_y[i>>2];
	}
	/* float --> m128 konverzio */
	__declspec(align(16)) __m128 sobel_x_l[FILTER_W*FILTER_H], sobel_y_l[FILTER_W*FILTER_H];
	for(int i=0; i<4*FILTER_W*FILTER_H; i++){
		sobel_x_l[i>>2].m128_f32[i&0x3]=sobel_x_ext[i];
		sobel_y_l[i>>2].m128_f32[i&0x3]=sobel_y_ext[i];
	}
	/* kepfeldolgozas */
	for(int row=imgFOffsetH; row<(imgHeight+imgFOffsetH); row++){
		for(int col=imgFOffsetW; col<(imgWidth+imgFOffsetW); col++){
			__declspec(align(16)) __m128 fval_x=const_0, fval_y=const_0, fval=const_0;
			/* konvolucio mindharom komponensre */
			for(int fy=0; fy<FILTER_H; fy++){
				int y=row+fy-imgFOffsetH;
				for(int fx=0; fx<FILTER_W; fx++){
					int x=col+fx-imgFOffsetW;
					__m128 pixel=_mm_load_ps(imgFloatSrc+(y*imgWidthF+x)*4);
					fval_x=_mm_add_ps(fval_x,_mm_mul_ps(sobel_x_l[fy*FILTER_W+fx],pixel));
					fval_y=_mm_add_ps(fval_y,_mm_mul_ps(sobel_y_l[fy*FILTER_W+fx],pixel));
				}
			}
			/* a ket eredmeny kombinalasa */
			fval=_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(fval_x,fval_x),_mm_mul_ps(fval_y,fval_y)));
			/* szurkearnyalatossa konvertalas */
			fval=_mm_mul_ps(const_rgb_to_gs,fval);
			fval=_mm_hadd_ps(fval,fval);
			fval=_mm_hadd_ps(fval,fval);
			/* kuszobozes */
			fval=_mm_cmpgt_ps(fval,thresh);
			fval=_mm_min_ps(fval,const_255);
			/* kimeneti pixel irasa (0...255 tartomanyba csonkolni) */
			_mm_stream_ps(imgFloatDst+(row*imgWidthF+col)*4,fval); //store is lehetne, de igy kikeruli a cache-t
		}
	}
}

void sobel_sse_omp(int imgHeight, int imgWidth, const float *imgFloatSrc, float *imgFloatDst){
	int imgWidthF=imgWidth+FILTER_W-1; //kiterjesztett kep szelessege
	int imgHeightF=imgHeight+FILTER_H-1; //kiterjesztett kep magassaga
	int imgFOffsetW=(FILTER_W-1)/2; //oldalso kiterjesztesek nagysaga
	int imgFOffsetH=(FILTER_H-1)/2; //also-felso kiterjesztesek nagysaga
	/* konstansok */
	__declspec(align(16)) __m128 const_0=_mm_set_ps(0.0, 0.0, 0.0, 0.0);
	__declspec(align(16)) __m128 const_255=_mm_set_ps(255.0, 255.0, 255.0, 255.0);
	__declspec(align(16)) __m128 const_rgb_to_gs=_mm_set_ps(0.0, 0.1140, 0.5870, 0.2989);
	__declspec(align(16)) __m128 thresh=_mm_set_ps(-1000, THRESHOLD, THRESHOLD, THRESHOLD);
	/* szuroegyutthatok negyszerezese */
	float sobel_x_ext[4*FILTER_W*FILTER_H], sobel_y_ext[4*FILTER_W*FILTER_H];
	for(int i=0; i<4*FILTER_W*FILTER_H; i++){
		sobel_x_ext[i]=filter_sobel_x[i>>2];
		sobel_y_ext[i]=filter_sobel_y[i>>2];
	}
	/* float --> m128 konverzio */
	__declspec(align(16)) __m128 sobel_x_l[FILTER_W*FILTER_H], sobel_y_l[FILTER_W*FILTER_H];
	for(int i=0; i<4*FILTER_W*FILTER_H; i++){
		sobel_x_l[i>>2].m128_f32[i&0x3]=sobel_x_ext[i];
		sobel_y_l[i>>2].m128_f32[i&0x3]=sobel_y_ext[i];
	}
	/* kepfeldolgozas */
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic,1) nowait
		for(int row=imgFOffsetH; row<(imgHeight+imgFOffsetH); row++){
			for(int col=imgFOffsetW; col<(imgWidth+imgFOffsetW); col++){
				__declspec(align(16)) __m128 fval_x=const_0, fval_y=const_0, fval=const_0;
				/* konvolucio mindharom komponensre */
				for(int fy=0; fy<FILTER_H; fy++){
					int y=row+fy-imgFOffsetH;
					for(int fx=0; fx<FILTER_W; fx++){
						int x=col+fx-imgFOffsetW;
						__m128 pixel=_mm_load_ps(imgFloatSrc+(y*imgWidthF+x)*4);
						fval_x=_mm_add_ps(fval_x,_mm_mul_ps(sobel_x_l[fy*FILTER_W+fx],pixel));
						fval_y=_mm_add_ps(fval_y,_mm_mul_ps(sobel_y_l[fy*FILTER_W+fx],pixel));
					}
				}
				/* a ket eredmeny kombinalasa */
				fval=_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(fval_x,fval_x),_mm_mul_ps(fval_y,fval_y)));
				/* szurkearnyalatossa konvertalas */
				fval=_mm_mul_ps(const_rgb_to_gs,fval);
				fval=_mm_hadd_ps(fval,fval);
				fval=_mm_hadd_ps(fval,fval);
				/* kuszobozes */
				fval=_mm_cmpgt_ps(fval,thresh);
				fval=_mm_min_ps(fval,const_255);
				/* kimeneti pixel irasa (0...255 tartomanyba csonkolni) */
				_mm_stream_ps(imgFloatDst+(row*imgWidthF+col)*4,fval); //store is lehetne, de igy kikeruli a cache-t
			}
		}
	}
}
