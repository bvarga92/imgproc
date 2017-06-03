#ifndef _SOBEL_H
#define _SOBEL_H_

	#define FILTER_W 5
	#define FILTER_H 5
	#define THRESHOLD 40

	/* Sobel-operator sima C megvalositasa. */
	void sobel(int imgHeight, int imgWidth, const float *imgFloatSrc, float *imgFloatDst);

	/* Sobel-operator OpenMP-vel tobbszalusitva. */
	void sobel_omp(int imgHeight, int imgWidth, const float *imgFloatSrc, float *imgFloatDst);

	/* Sobel-operator SIMD utasitasokkal. */
	void sobel_sse(int imgHeight, int imgWidth, const float *imgFloatSrc, float *imgFloatDst);

	/* Sobel-operator SIMD utasitasokkal es OpenMP parhuzamositassal. */
	void sobel_sse_omp(int imgHeight, int imgWidth, const float *imgFloatSrc, float *imgFloatDst);
				 
#endif
