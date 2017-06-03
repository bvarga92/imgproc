#ifndef _CONV_FILTER_H_
#define _CONV_FILTER_H_

	#define FILTER_W 5
	#define FILTER_H 5

	extern const float filter_laplace[25]; //2D Laplace-operator
	extern const float filter_sobel_x[25]; //Sobel-operator x iranyban
	extern const float filter_sobel_y[25]; //Sobel-operator y iranyban
	extern const float filter_horiz_edges[25]; //vizszintes elek
	extern const float filter_vert_edges[25]; //fuggoleges elek
	extern const float filter_average[25]; //atlagolo szuro
	extern const float filter_gauss_lpf[25]; //Gauss-eletlenites
	extern const float filter_highpass[25]; //felulatereszto szuro elesiteshez
	extern const float filter_motionblur[25]; //bemozditas atlosan
	extern const float filter_emboss[25]; //dombornyomas
	extern const float filter_emboss2[25]; //eros dombornyomas

	/* 2D konvolucio sima C megvalositasa. */
	void conv_filter(int imgHeight, int imgWidth, const float *filter, const float *imgFloatSrc, float *imgFloatDst);

	/* 2D konvolucio OpenMP-vel tobbszalusitva. */
	void conv_filter_omp(int imgHeight, int imgWidth, const float *filter, const float *imgFloatSrc, float *imgFloatDst);

	/* 2D konvolucio SIMD utasitasokkal. */
	void conv_filter_sse(int imgHeight, int imgWidth, const float *filter, const float *imgFloatSrc, float *imgFloatDst);

	/* 2D konvolucio SIMD utasitasokkal es OpenMP parhuzamositassal. */
	void conv_filter_sse_omp(int imgHeight, int imgWidth, const float *filter, const float *imgFloatSrc, float *imgFloatDst);
				 
#endif
