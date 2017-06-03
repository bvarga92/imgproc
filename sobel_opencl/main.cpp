/***************************************************************************************************************/
/* Visual Studio projekt beallitasok:                                                                          */
/*   - a Configuration Maganerben adjunk hozza egy x64 platformot (Win32 masolasaval)                          */
/*   - C/C++ --> General --> Additional Include Directories: .\DevIL_x64\include\;$(AMDAPPSDKROOT)/include     */
/*   - Linker --> General --> Additional Library Directories: .\DevIL_x64;$(AMDAPPSDKROOT)/lib/x86_64          */
/*   - Linker --> Input --> Additional Dependencies: OpenCL.lib;DevIL.lib;ILU.lib;ILUT.lib;                    */
/* A .dll fajlokat be kell masolni oda, ahova a leforditott .exe kerul (megfelelo Debug vagy Release mappa).   */
/* Csak szines, 24 bites szinmelysegu kepeket hasznaljunk!                                                     */
/* Idomeres eredmenye: 178.5 Mpixel/s                                                                          */
/***************************************************************************************************************/

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <IL/ilut.h>
#include <IL/ilu.h>
#include <CL\cl.h>

#define RUNS 10
#define FILTER_W 5
#define FILTER_H 5
#define INPUTFILE   "_images/img1.jpg"
#define OUTPUTFILE  "_images/img1_sobel.jpg"
#define KERNELFILE  "_src/kernel_32x8.cl"
#define TB_X 32
#define TB_Y  8

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

void sobelOpenCL(int imgHeight, int imgWidth, unsigned char *imgSrc, unsigned char *imgDst){
	cl_platform_id *platforms, platform;
	cl_device_id *devices, device;
	cl_uint numPlatforms, numDevices;
	size_t size, globalWorkSize[2], localWorkSize[2];
	char *str;
	cl_context context;
	FILE *fp;
	cl_program program;
	cl_kernel kernel;
	cl_mem device_imgSrc, device_imgDst, device_coeffsX, device_coeffsY;
	cl_command_queue queue;
	cl_event events[RUNS];
	cl_ulong timeStart, timeEnd;
	int imgWidthF=imgWidth+FILTER_W-1;
	int imgHeightF=imgHeight+FILTER_H-1;
	/* platform */
	clGetPlatformIDs(0,NULL,&numPlatforms);
	if(numPlatforms==0){
		printf("Error: no platforms found.\n");
		exit(1);
	}
	platforms=(cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
	clGetPlatformIDs(numPlatforms,platforms,NULL);
	platform=platforms[0];
	free(platforms);
	clGetPlatformInfo(platform,CL_PLATFORM_NAME,NULL,NULL,&size);
	str=(char*)malloc(size);
	clGetPlatformInfo(platform,CL_PLATFORM_NAME,size,str,NULL);
	printf("Using platform: %s\n",str);
	free(str);
	/* device */
	clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&numDevices);
	if(numDevices==0){
		printf("Error: no devices found.\n");
		exit(1);
	}
	devices=(cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
	clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,numDevices,devices,NULL);
	device=devices[0];
	free(devices);
	clGetDeviceInfo(device,CL_DEVICE_NAME,NULL,NULL,&size);
	str=(char*)malloc(size);
	clGetDeviceInfo(device,CL_DEVICE_NAME,size,str,NULL);
	printf("Using device: %s\n",str);
	free(str);
	/* kernel program */
	context=clCreateContext(NULL,1,&device,NULL,NULL,NULL);
	fp=fopen(KERNELFILE,"rb");
	if(!fp){
		printf("Failed to open kernel source file.\n");
		exit(1);
	}
	fseek(fp,0,SEEK_END);
	size=(size_t)ftell(fp);
	rewind(fp);
	str=(char*)malloc(size);
	fread(str,1,size,fp);
	fclose(fp);
	program=clCreateProgramWithSource(context,1,(const char**)&str,(const size_t*)&size,NULL);
	if(clBuildProgram(program,1,&device,NULL,NULL,NULL)!=CL_SUCCESS){
		clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,NULL,NULL,&size);
		str=(char*)realloc(str,size);
		clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,size,str,NULL);
		printf("Error building kernel: %s",str);
		free(str);
		exit(1);
	}
	free(str);
	kernel=clCreateKernel(program,"sobel_kernel",NULL);
	/* kernel parameterek atadasa */
	device_imgSrc=clCreateBuffer(context,CL_MEM_READ_ONLY,imgHeightF*imgWidthF*3,NULL,NULL);
	device_imgDst=clCreateBuffer(context,CL_MEM_WRITE_ONLY,imgHeightF*imgWidthF*3,NULL,NULL);
	device_coeffsX=clCreateBuffer(context,CL_MEM_READ_ONLY,FILTER_W*FILTER_H*sizeof(float),NULL,NULL);
	device_coeffsY=clCreateBuffer(context,CL_MEM_READ_ONLY,FILTER_W*FILTER_H*sizeof(float),NULL,NULL);
	queue=clCreateCommandQueue(context,device,CL_QUEUE_PROFILING_ENABLE,NULL);
	clEnqueueWriteBuffer(queue,device_imgSrc,CL_TRUE,0,imgHeightF*imgWidthF*3,imgSrc,0,NULL,NULL);
	clEnqueueWriteBuffer(queue,device_coeffsX,CL_TRUE,0,FILTER_W*FILTER_H*sizeof(float),filter_sobel_x,0,NULL,NULL);
	clEnqueueWriteBuffer(queue,device_coeffsY,CL_TRUE,0,FILTER_W*FILTER_H*sizeof(float),filter_sobel_y,0,NULL,NULL);
	clSetKernelArg(kernel,0,sizeof(device_imgSrc),&device_imgSrc);
	clSetKernelArg(kernel,1,sizeof(device_imgDst),&device_imgDst);
	clSetKernelArg(kernel,2,sizeof(device_coeffsX),&device_coeffsX);
	clSetKernelArg(kernel,3,sizeof(device_coeffsY),&device_coeffsY);
	clSetKernelArg(kernel,4,sizeof(int),&imgWidthF);
	clFinish(queue);
	/* kernel futtatasa */
	globalWorkSize[0]=imgWidth;
	globalWorkSize[1]=imgHeight;
	localWorkSize[0]=TB_X;
	localWorkSize[1]=TB_Y;
	for(unsigned i=0;i<RUNS;i++) clEnqueueNDRangeKernel(queue,kernel,2,NULL,globalWorkSize,localWorkSize,0,NULL,&events[i]);
	clWaitForEvents(1,&events[RUNS-1]);
	clGetEventProfilingInfo(events[0],CL_PROFILING_COMMAND_START,sizeof(timeStart),&timeStart,NULL);
	clGetEventProfilingInfo(events[RUNS-1],CL_PROFILING_COMMAND_END,sizeof(timeEnd),&timeEnd,NULL);
	printf("Total kernel time: %6.4f ms (%d runs)\n",(timeEnd-timeStart)/(1000000.0),RUNS);
	printf("Mpixel/s: %4.4f\n",RUNS*imgWidth*imgHeight*1000.0/(timeEnd-timeStart));
	/* eredmeny kiolvasasa */
	clEnqueueReadBuffer(queue,device_imgDst,CL_TRUE,0,imgHeightF*imgWidthF*3,imgDst,0,NULL,NULL);
	/* befejezes */
	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(device_imgSrc);
	clReleaseMemObject(device_imgDst);
	clReleaseMemObject(device_coeffsX);
	clReleaseMemObject(device_coeffsY);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
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
	int imgWidth=ilGetInteger(IL_IMAGE_WIDTH);
	int imgHeight=ilGetInteger(IL_IMAGE_HEIGHT);
	ILint imgOrigin=ilGetInteger(IL_ORIGIN_MODE);
	int imgWidthF=imgWidth+FILTER_W-1;
	int imgHeightF=imgHeight+FILTER_H-1;
	int imgFOffsetW=(FILTER_W-1)/2;
	int imgFOffsetH=(FILTER_H-1)/2;
	printf("Input resolution: %4dx%4d\n",imgWidth,imgHeight);
	/* tomb az uj kepnek, kinullazzuk */
	unsigned char *imgSrc=(unsigned char*)(_aligned_malloc(3*imgWidthF*imgHeightF*sizeof(unsigned char),32));
	for(int row=0; row<imgHeightF; row++)
		for(int col=0; col<imgWidthF;col++){
			int pixel=(row*imgWidthF+col)*3;
			imgSrc[pixel+0]=0.0f;
			imgSrc[pixel+1]=0.0f;
			imgSrc[pixel+2]=0.0f;
		}
	/* atmasoljuk a kepet a kiterjesztett tombbe */
	for(int row=0; row<imgHeight; row++)
		for(int col=0; col<imgWidth; col++){
			int pixel_dst=((row+imgFOffsetH)*imgWidthF+(col+imgFOffsetW))*3;
			int pixel_src=(row*imgWidth+col)*3;
			imgSrc[pixel_dst+0]=(unsigned char)imgData[pixel_src+0]; //r
			imgSrc[pixel_dst+1]=(unsigned char)imgData[pixel_src+1]; //g
			imgSrc[pixel_dst+2]=(unsigned char)imgData[pixel_src+2]; //b
		}
	/* kepfeldolgozas */
	unsigned char *imgRes=(unsigned char*)_aligned_malloc(4*imgWidthF*imgHeightF*sizeof(unsigned char),32);
	sobelOpenCL(imgHeight, imgWidth, imgSrc, imgRes);
	/* eredmeny kiirasa */
	for(int row=0; row<imgHeight; row++)
		for(int col=0; col<imgWidth; col++){
			int pixel_dst=(row*imgWidth+col)*3;
			int pixel_src=((row+imgFOffsetH)*imgWidthF+(col+imgFOffsetW))*3;
			imgData[pixel_dst+0]=(ILubyte)imgRes[pixel_src+0];
			imgData[pixel_dst+1]=(ILubyte)imgRes[pixel_src+1];
			imgData[pixel_dst+2]=(ILubyte)imgRes[pixel_src+2];
		}
	_aligned_free(imgSrc);
	_aligned_free(imgRes);
	ilSetData(imgData);
	ilEnable(IL_FILE_OVERWRITE);
	ilSaveImage((const wchar_t*)OUTPUTFILE);
	ilDeleteImages(1,&ilImg);
	return 0;
}
