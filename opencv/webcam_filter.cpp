#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

float ident[3][3]={{0,0,0},{0,1,0},{0,0,0}};
float emboss[3][3]={{-2,-1,0},{-1,1,1},{0,1,2}};
float outline[3][3]={{-1,-1,-1},{-1,8,-1},{-1,-1,-1}};
float sharpen[3][3]={{0,-1,0},{-1,5,-1},{0,-1,0}};
float avgblur[5][5]={{0,0,0.0769f,0,0},{0,0.0769f,0.0769f,0.0769f,0},{0.0769f,0.0769f,0.0769f,0.0769f,0.0769f},{0,0.0769f,0.0769f,0.0769f,0},{0,0,0.0769f,0,0}};

int main(){
	Mat in, gs, out;
	Mat filter=Mat(3,3,CV_32FC(1),ident);
	VideoCapture vid=VideoCapture(0);
	int key=0;
	while(key!=27){
		vid>>in;
		cvtColor(in,gs,CV_RGB2GRAY);
		filter2D(gs,out,-1,filter);
		imshow("Webcam Capture",out);
		key=waitKey(10);
		switch(key){
			case 49: filter=Mat(3,3,CV_32FC(1),ident); break;
			case 50: filter=Mat(3,3,CV_32FC(1),emboss); break;
			case 51: filter=Mat(3,3,CV_32FC(1),outline); break;
			case 52: filter=Mat(3,3,CV_32FC(1),sharpen); break;
			case 53: filter=Mat(5,5,CV_32FC(1),avgblur); break;
		}
	}
	return 0;
}
