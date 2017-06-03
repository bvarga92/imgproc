#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THRESHOLD 40

__global__ void sobel_kernel_16x16(unsigned char* gInput, unsigned char* gOutput, float *coeffsX, float *coeffsY, int imgWidthF){
	int row     = blockIdx.y*blockDim.y+threadIdx.y+2;
	int col     = blockIdx.x*blockDim.x+threadIdx.x+2;
	int thid_1d = threadIdx.y*blockDim.x+threadIdx.x;
	int ld_rgb  = thid_1d %3;
	int ld_col  = (thid_1d/3)%20;
	int ld_row  = thid_1d/60;
	int ld_base = blockIdx.y*blockDim.y*3*imgWidthF + blockIdx.x*blockDim.x*3 + ld_row*3* imgWidthF;
	int out_pix = (row*imgWidthF+col)*3;
	/* betoltes */
	__shared__ float in_shmem[20][20][3];
	if(thid_1d<240){ //3*20*4
		#pragma unroll
		for(int ld=0;ld<5;ld++){
			in_shmem[ld_row+ld*4][ld_col][ld_rgb]=(float)gInput[ld_base+(thid_1d%60)];
			ld_base=ld_base+imgWidthF*3*4;
		}
	}
	__syncthreads();
	/* konvolucio */
	float accX[3]={0.0f,0.0f,0.0f}, accY[3]={0.0f,0.0f,0.0f};
	#pragma unroll
	for(int fy=0;fy<5;fy++){
		#pragma unroll
		for(int fx=0;fx<5;fx++){
			float coeffX=coeffsX[fy*5+fx];
			float coeffY=coeffsY[fy*5+fx];
			#pragma unroll
			for(int rgb=0;rgb<3;rgb++){
				float pix=in_shmem[threadIdx.y+fy][threadIdx.x+fx][rgb];
				accX[rgb]=fmaf(coeffX,pix,accX[rgb]);
				accY[rgb]=fmaf(coeffY,pix,accY[rgb]);
			}
		}
	}
	/* a ket eredmeny kombinalasa es szurkearnyalatossa konvertalas */
	float gs=0.2989f*hypotf(accX[0],accY[0])+0.5870f*hypotf(accX[1],accY[1])+0.1140f*hypotf(accX[2],accY[2]);
	/* kuszobozes */
	gs=(gs>THRESHOLD)*255.0f;
	#pragma unroll
	for(int rgb=0;rgb<3;rgb++) gOutput[out_pix+rgb]=(unsigned char)gs;
}

__global__ void sobel_kernel_32x8(unsigned char* gInput, unsigned char* gOutput, float *coeffsX, float *coeffsY, int imgWidthF){
	int row     = blockIdx.y*blockDim.y+threadIdx.y+2;
	int col     = blockIdx.x*blockDim.x+threadIdx.x+2;
	int thid_1d = threadIdx.y*blockDim.x+threadIdx.x;
	int ld_rgb  = thid_1d %3;
	int ld_col  = (thid_1d/3)%36;
	int ld_row  = thid_1d/108;
	int ld_base = blockIdx.y*blockDim.y*3*imgWidthF + blockIdx.x*blockDim.x*3 + ld_row*3* imgWidthF;
	int out_pix = (row*imgWidthF+col)*3;
	/* betoltes */
	__shared__ float in_shmem[12][36][3];
	if(thid_1d<216){ //3*36*2
		#pragma unroll
		for(int ld=0;ld<6;ld++){
			in_shmem[ld_row+ld*2][ld_col][ld_rgb]=(float)gInput[ld_base+(thid_1d%108)];
			ld_base=ld_base+imgWidthF*3*2;
		}
	}
	__syncthreads();
	/* konvolucio */
	float accX[3]={0.0f,0.0f,0.0f}, accY[3]={0.0f,0.0f,0.0f};
	#pragma unroll
	for(int fy=0;fy<5;fy++){
		#pragma unroll
		for(int fx=0;fx<5;fx++){
			float coeffX=coeffsX[fy*5+fx];
			float coeffY=coeffsY[fy*5+fx];
			#pragma unroll
			for(int rgb=0;rgb<3;rgb++){
				float pix=in_shmem[threadIdx.y+fy][threadIdx.x+fx][rgb];
				accX[rgb]=fmaf(coeffX,pix,accX[rgb]);
				accY[rgb]=fmaf(coeffY,pix,accY[rgb]);
			}
		}
	}
	/* a ket eredmeny kombinalasa es szurkearnyalatossa konvertalas */
	float gs=0.2989f*hypotf(accX[0],accY[0])+0.5870f*hypotf(accX[1],accY[1])+0.1140f*hypotf(accX[2],accY[2]);
	/* kuszobozes */
	gs=(gs>THRESHOLD)*255.0f;
	#pragma unroll
	for(int rgb=0;rgb<3;rgb++) gOutput[out_pix+rgb]=(unsigned char)gs;
}
