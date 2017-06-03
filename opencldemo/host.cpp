#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <streambuf>
#include <cstdlib>
#include <CL/cl.hpp>

using namespace std;

const unsigned numElements=100; //vektorok elemszama

int main(void){
	/* platform */
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if(platforms.size()==0){
		cout<<"Error: no platforms found."<<endl;
		return 1;
	}
	for(unsigned i=0;i<platforms.size();i++) cout<<"Platform #"<<i+1<<": "<<platforms[i].getInfo<CL_PLATFORM_NAME>()<<endl;
	cout<<"Select platform: ";
	unsigned ip;
	cin>>ip;
	if(ip==0 || ip>platforms.size()) return 1;
	cout<<"Using platform #"<<ip<<"."<<endl<<endl;
	cl::Platform platform=platforms[ip-1];
	/* device */
	vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL,&devices);
	if(devices.size()==0){
		cout<<"Error: no devices found."<<endl;
		return 1;
	}
	for(unsigned i=0;i<devices.size();i++) cout<<"Device #"<<i+1<<": "<<devices[i].getInfo<CL_DEVICE_NAME>()<<endl;
	cout<<"Select device: ";
	unsigned id;
	cin>>id;
	if(id==0 || id>devices.size()) return 1;
	cout<<"Using device #"<<id<<"."<<endl<<endl;
	cl::Device device=devices[id-1];
	/* kernel program */
	cl::Context context({device});
	cl::Program::Sources sources;
	ifstream file("kernel.cl");
	if((!file.is_open())||(!file.good())){
		cout<<"Failed to open kernel source file."<<endl;
		return 1;
	}
	string code((istreambuf_iterator<char>(file)),istreambuf_iterator<char>());
	sources.push_back({code.c_str(),code.length()});
	cl::Program program(context,sources);
	if(program.build({device})!=CL_SUCCESS){
		cout<<"Error building kernel: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)<<endl;
		return 1;
	}
	cl::Kernel kernel=cl::Kernel(program,"vecadd");
	/* kernel parameterek atadasa */
	int a[numElements], b[numElements];
	srand(123);
	for(unsigned i=0;i<numElements;i++){
		a[i]=rand()%1000;
		b[i]=rand()%1000;
	}
	cl::Buffer bufferA(context,CL_MEM_READ_ONLY,sizeof(int)*numElements);
	cl::Buffer bufferB(context,CL_MEM_READ_ONLY,sizeof(int)*numElements);
	cl::Buffer bufferC(context,CL_MEM_READ_WRITE,sizeof(int)*numElements);
	cl::CommandQueue queue(context,device);
	queue.enqueueWriteBuffer(bufferA,CL_TRUE,0,sizeof(int)*numElements,a);
	queue.enqueueWriteBuffer(bufferB,CL_TRUE,0,sizeof(int)*numElements,b);
	kernel.setArg(0,bufferA);
	kernel.setArg(1,bufferB);
	kernel.setArg(2,bufferC);
	/* kernel futtatasa numElements darab szallal */
	cl::NDRange globalWorkSize(numElements);
	queue.enqueueNDRangeKernel(kernel,cl::NullRange,globalWorkSize,cl::NullRange);
	queue.finish();
	/* eredmeny kiolvasasa */
	int c[numElements];
	queue.enqueueReadBuffer(bufferC,CL_TRUE,0,sizeof(int)*numElements,c);
	cout<<endl;
	for(unsigned i=0;i<numElements;i++) cout<<setw(3)<<a[i]<<" + "<<setw(3)<<b[i]<<" = "<<setw(4)<<c[i]<<endl;
	cin.get();
	cin.get();
	return 0;
}
