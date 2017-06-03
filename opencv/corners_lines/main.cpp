#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){
	/* kep betoltese */
	Mat im=imread("input.jpg",CV_LOAD_IMAGE_COLOR);
	if(!im.data){
		cout<<"Failed to open image."<<endl;
		return 1;
	}
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Input",im);
	/* szurkearnyalatossa konvertalas */
	Mat gray;
	cvtColor(im,gray,CV_BGR2GRAY);
	/* eldetektalas Canny-algoritmussal */
	Mat edges;
	Canny(gray,edges,50,150,3);
	/* egyenesek illesztese Hough-transzformacioval */
	vector<Vec2f> lines;
	HoughLines(edges,lines,1,CV_PI/180.0f,135);
	for(unsigned i=0;i<lines.size();i++){
		double a=cos(lines[i][1]);
		double b=sin(lines[i][1]);
		double x0=a*lines[i][0];
		double y0=b*lines[i][0];
		Point pt1(cvRound(x0-1000*b),cvRound(y0+1000*a));
		Point pt2(cvRound(x0+1000*b),cvRound(y0-1000*a));
		line(im,pt1,pt2,Scalar(0,0,255));
	}
	/* sarkok detektalasa Shi-Tomasi-algoritmussal */
	vector<Point2d> corners;
	goodFeaturesToTrack(gray,corners,80,0.1,30);
	for(unsigned i=0;i<corners.size();i++) circle(im,corners[i],4,Scalar(0,255,0),-1);
	/* az eredmeny megjelenitese es fajlba irasa */
	imwrite("output.jpg",im);
	namedWindow("Output",WINDOW_AUTOSIZE);
	imshow("Output",im);
	waitKey(0);
	return 0;
}
