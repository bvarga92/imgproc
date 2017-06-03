#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define NUM_ELEMENTS 100 //vektorok elemszama

int main(void){
	int a[NUM_ELEMENTS], b[NUM_ELEMENTS], c[NUM_ELEMENTS];
	cl_platform_id *platforms, platform;
	cl_device_id *devices, device;
	cl_uint numPlatforms, numDevices;
	size_t size, globalWorkSize;
	char *str;
	cl_context context;
	FILE *fp;
	cl_program program;
	cl_kernel kernel;
	cl_mem bufferA, bufferB, bufferC;
	cl_command_queue queue;
	unsigned i;
	/* platform */
	clGetPlatformIDs(0,NULL,&numPlatforms);
	if(numPlatforms==0){
		printf("Error: no platforms found.\n");
		return 1;
	}
	platforms=(cl_platform_id*)malloc(sizeof(cl_platform_id)*numPlatforms);
	clGetPlatformIDs(numPlatforms,platforms,NULL);
	for(i=0;i<numPlatforms;i++){
		clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,NULL,NULL,&size);
		str=(char*)malloc(size);
		clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,size,str,NULL);
		printf("Platform #%u: %s\n",i+1,str);
		free(str);
	}
	printf("Select platform: ");
	scanf("%u",&i);
	if(i==0 || i>numPlatforms) return 1;
	printf("Using platform #%u.\n\n",i);
	platform=platforms[i-1];
	free(platforms);
	/* device */
	clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,NULL,&numDevices);
	if(numDevices==0){
		printf("Error: no devices found.\n");
		return 1;
	}
	devices=(cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
	clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,numDevices,devices,NULL);
	for(i=0;i<numDevices;i++){
		clGetDeviceInfo(devices[i],CL_DEVICE_NAME,NULL,NULL,&size);
		str=(char*)malloc(size);
		clGetDeviceInfo(devices[i],CL_DEVICE_NAME,size,str,NULL);
		printf("Device #%u: %s\n",i+1,str);
		free(str);
	}
	printf("Select device: ");
	scanf("%u",&i);
	if(i==0 || i>numDevices) return 1;
	printf("Using device #%u.\n\n",i);
	device=devices[i-1];
	free(devices);
	/* kernel program */
	context=clCreateContext(NULL,1,&device,NULL,NULL,NULL);
	fp=fopen("kernel.cl","rb");
	if(!fp){
		printf("Failed to open kernel source file.\n");
		return 1;
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
		return 1;
	}
	free(str);
	kernel=clCreateKernel(program,"vecadd",NULL);
	/* kernel parameterek atadasa */
	srand(123);
	for(i=0;i<NUM_ELEMENTS;i++){
		a[i]=rand()%1000;
		b[i]=rand()%1000;
	}
	bufferA=clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(int)*NUM_ELEMENTS,NULL,NULL);
	bufferB=clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(int)*NUM_ELEMENTS,NULL,NULL);
	bufferC=clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(int)*NUM_ELEMENTS,NULL,NULL);
	queue=clCreateCommandQueue(context,device,NULL,NULL);
	clEnqueueWriteBuffer(queue,bufferA,CL_TRUE,0,sizeof(int)*NUM_ELEMENTS,a,0,NULL,NULL);
	clEnqueueWriteBuffer(queue,bufferB,CL_TRUE,0,sizeof(int)*NUM_ELEMENTS,b,0,NULL,NULL);
	clSetKernelArg(kernel,0,sizeof(cl_mem),&bufferA);
	clSetKernelArg(kernel,1,sizeof(cl_mem),&bufferB);
	clSetKernelArg(kernel,2,sizeof(cl_mem),&bufferC);
	/* kernel futtatasa NUM_ELEMENTS darab szallal */
	globalWorkSize=(size_t)NUM_ELEMENTS;
	clEnqueueNDRangeKernel(queue,kernel,1,NULL,&globalWorkSize,NULL,0,NULL,NULL);
	clFinish(queue);
	/* eredmeny kiolvasasa */
	clEnqueueReadBuffer(queue,bufferC,CL_TRUE,0,sizeof(int)*NUM_ELEMENTS,c,0,NULL,NULL);
	printf("\n");
	for(i=0;i<NUM_ELEMENTS;i++) printf("%3d + %3d = %4d\n",a[i],b[i],c[i]);
	/* befejezes */
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	getchar();
	getchar();
	return 0;
}
