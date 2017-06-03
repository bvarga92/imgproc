#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int minRadius=15;
int tailLength=10;
int hmin=80;
int hmax=155;
int smin=73;
int smax=255;
int vmin=47;
int vmax=255;

Scalar lower=Scalar(hmin,smin,vmin);
Scalar upper=Scalar(hmax,smax,vmax);

void controlsCallback(int,void*){
	lower.val[0]=getTrackbarPos("H min.","Color Control");
	lower.val[1]=getTrackbarPos("S min.","Color Control");
	lower.val[2]=getTrackbarPos("V min.","Color Control");
	upper.val[0]=getTrackbarPos("H max.","Color Control");
	upper.val[1]=getTrackbarPos("S max.","Color Control");
	upper.val[2]=getTrackbarPos("V max.","Color Control");
}

int main(){
	VideoCapture vid=VideoCapture(0);
	if(!vid.isOpened()){
		cout<<"Connecting to camera failed."<<endl;
		return 1;
	}
	namedWindow("Color Control",WINDOW_KEEPRATIO);
	createTrackbar("H min.","Color Control",&hmin,255,controlsCallback);
	createTrackbar("H max.","Color Control",&hmax,255,controlsCallback);
	createTrackbar("S min.","Color Control",&smin,255,controlsCallback);
	createTrackbar("S max.","Color Control",&smax,255,controlsCallback);
	createTrackbar("V min.","Color Control",&vmin,255,controlsCallback);
	createTrackbar("V max.","Color Control",&vmax,255,controlsCallback);
	namedWindow("Object Tracking",WINDOW_AUTOSIZE);
	Mat in, out;
	vector<Point2f> tail(tailLength);
	unsigned t=0, l=0;
	bool showMask=false;
	while(true){
		vid>>in;
		/* elofeldolgozas: alulatereszto szures, szin maszkolasa, erozio-dilatacio */
		GaussianBlur(in,out,Size(15,15),0);
		cvtColor(out,out,COLOR_BGR2HSV);
		inRange(out,lower,upper,out);
		erode(out,out,Mat(),Point(-1,-1),2);
		dilate(out,out,Mat(),Point(-1,-1),2);
		/* megkeressuk a maszkolt targy konturjat */
		vector<vector<Point>> contours;
		findContours(out.clone(),contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
		if(contours.size()>0){
			/* ha tobb konturt is talaltunk, megkeressuk a legnagyobb teruletut */
			unsigned largest=0;
			double max=0;
			for(unsigned i=0;i<contours.size();i++){
				if(contourArea(contours[i])>max){
					max=contourArea(contours[i]);
					largest=i;
				}
			}
			/* megkeressuk a kontur kore irhato kort, es rarajzoljuk a kepre */
			Point2f c;
			float r;
			minEnclosingCircle(contours[largest],c,r);
			if(r>=minRadius){
				circle(in,c,r,Scalar(0,0,255),2);
				circle(in,c,3,Scalar(0,0,255),-1);
				tail[t]=c;
				t=(t+1)%tailLength;
				if(l<tailLength-1) l=t;
			}
		}
		/* megrajzoljuk a csovat */
		for(unsigned i=0;i<l;i++)
			line(in,tail[(t+i)%tailLength],tail[(t+i+1)%tailLength],Scalar(0,0,255),2);
		/* az 1-es billentyu hatasara a maszkot mutatjuk, a 2-essel visszavaltunk */
		int key=waitKey(1);
		if(key==27)
			break;
		else if(key==49)
			showMask=true;
		else if(key==50)
			showMask=false;
		imshow("Object Tracking",showMask?out:in);
	}
	return 0;
}
