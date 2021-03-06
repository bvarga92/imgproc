typedef struct{int x, y;} cuda_id;

__kernel void conv_kernel(__global unsigned char* gInput, __global unsigned char* gOutput, __constant float *coeffs, int imgWidthF){
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
	int ld_col  = (thid_1d/3)%36;
	int ld_row  = thid_1d/108;
	int ld_base = blockIdx.y*blockDim.y*3*imgWidthF + blockIdx.x*blockDim.x*3 + ld_row*3*imgWidthF;
	int out_pix = (row*imgWidthF+col)*3;
	/* betoltes */
	__local float in_shmem[12][36][3];
	if(thid_1d<216){ //3*36*2
		#pragma unroll
		for(int ld=0;ld<6;ld++){
			in_shmem[ld_row+ld*2][ld_col][ld_rgb]=(float)gInput[ld_base+(thid_1d%108)];
			ld_base+=imgWidthF*3*2;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	/* konvolucio */
	float4 acc=(float4)(0.0f);
	#pragma unroll
	for(int fy=0;fy<5;fy++){
		#pragma unroll
		for(int fx=0;fx<5;fx++){
			float4 coeff;
			coeff=(float4)(coeffs[fy*5+fx]);
			float4 pix;
			pix.s0=in_shmem[threadIdx.y+fy][threadIdx.x+fx][0]; //r
			pix.s1=in_shmem[threadIdx.y+fy][threadIdx.x+fx][1]; //g
			pix.s2=in_shmem[threadIdx.y+fy][threadIdx.x+fx][2]; //b
			pix.s3=0.0f;
			acc=mad(coeff,pix,acc);
		}
	}
	/* 0...255 tartomanyba csonkolas */
	acc=clamp(acc,0.0f,255.0f);
	/* kimeneti pixel irasa */
	gOutput[out_pix+0]=(unsigned char)acc.s0;
	gOutput[out_pix+1]=(unsigned char)acc.s1;
	gOutput[out_pix+2]=(unsigned char)acc.s2;
}
