#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){
	VideoCapture vid=VideoCapture(0);
	if(!vid.isOpened()){
		cout<<"Connecting to camera failed."<<endl;
		return 1;
	}
	namedWindow("Motion Detection",WINDOW_AUTOSIZE);
	Mat in, shot[3], d[2], out;
	for(unsigned i=0;i<3;i++){
		vid>>in;
		cvtColor(in,shot[i],COLOR_RGB2GRAY);
	}
	while(waitKey(10)!=27){
		absdiff(shot[2],shot[1],d[0]);
		absdiff(shot[1],shot[0],d[1]);
		bitwise_and(d[0],d[1],out);
		threshold(out,out,30,255,THRESH_BINARY);
		if(countNonZero(out)>100) putText(out,"ALARM!",Point(10,50),FONT_HERSHEY_DUPLEX,1.5,Scalar(255,255,255),2);
		imshow("Motion Detection",out);
		shot[0]=shot[1].clone();
		shot[1]=shot[2].clone();
		vid>>in;
		cvtColor(in,shot[2],COLOR_RGB2GRAY);
	}
	return 0;
}
