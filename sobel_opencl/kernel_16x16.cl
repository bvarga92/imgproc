#define THRESHOLD 40

typedef struct{int x, y;} cuda_id;

__kernel void sobel_kernel(__global unsigned char* gInput, __global unsigned char* gOutput, __constant float *coeffsX, __constant float *coeffsY, int imgWidthF){
	/* cimek szamitasa */
	cuda_id threadIdx, blockIdx, blockDim;
	threadIdx.x = get_local_id(0);
	threadIdx.y = get_local_id(1);
	blockIdx.x  = get_group_id(0);
	blockIdx.y  = get_group_id(1);
	blockDim.x  = get_local_size(0);
	blockDim.y  = get_local_size(1);
	int row     = blockIdx.y*blockDim.y+threadIdx.y+2;
	int col     = blockIdx.x*blockDim.x+threadIdx.x+2;
	int thid_1d = threadIdx.y*blockDim.x+threadIdx.x;
	int ld_rgb  = thid_1d%3;
	int ld_col  = (thid_1d/3)%20;
	int ld_row  = thid_1d/60;
	int ld_base = blockIdx.y*blockDim.y*3*imgWidthF + blockIdx.x*blockDim.x*3 + ld_row*3*imgWidthF;
	int out_pix = (row*imgWidthF+col)*3;
	/* betoltes */
	__local float in_shmem[20][20][3];
	if(thid_1d<240){ //3*20*4
		#pragma unroll
		for(int ld=0;ld<5;ld++){
			in_shmem[ld_row+ld*4][ld_col][ld_rgb]=(float)gInput[ld_base+(thid_1d%60)];
			ld_base+=imgWidthF*3*4;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	/* konvolucio */
	float8 acc=(float8)(0.0f);
	#pragma unroll
	for(int fy=0;fy<5;fy++){
		#pragma unroll
		for(int fx=0;fx<5;fx++){
			float8 coeff;
			coeff.lo=(float4)(coeffsX[fy*5+fx]);
			coeff.hi=(float4)(coeffsY[fy*5+fx]);
			float8 pix;
			pix.s0=in_shmem[threadIdx.y+fy][threadIdx.x+fx][0]; //r
			pix.s1=in_shmem[threadIdx.y+fy][threadIdx.x+fx][1]; //g
			pix.s2=in_shmem[threadIdx.y+fy][threadIdx.x+fx][2]; //b
			pix.s3=0.0f;
			pix.hi=pix.lo;
			acc=mad(coeff,pix,acc);
		}
	}
	/* a ket eredmeny kombinalasa */
	float4 pix_data=hypot(acc.lo,acc.hi);
	/* szurkearnyalatossa konvertalas */
	float gs=dot(pix_data,(float4)(0.2989f,0.5870f,0.1140f,0.0f));
	/* kuszobozes */
	gs=255*step(THRESHOLD,gs);
	/* kimeneti pixel irasa */
	#pragma unroll
	for(int rgb=0;rgb<3;rgb++) gOutput[out_pix+rgb]=(unsigned char)gs;
}
