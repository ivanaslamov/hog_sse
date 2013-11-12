
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>  
#include <sys/time.h>
#include <ctime>

#include "hog.h"

using namespace std;
using namespace cv;

int main(int, char**)
{
	/* Linux */
	struct timeval tv;

	Mat src = imread("leonid.jpg",0);

	Mat bin0 = Mat(src.rows, src.cols, CV_8U);
	Mat bin1 = Mat(src.rows, src.cols, CV_8U);
	Mat bin2 = Mat(src.rows, src.cols, CV_8U);
	Mat bin3 = Mat(src.rows, src.cols, CV_8U);
	Mat bin4 = Mat(src.rows, src.cols, CV_8U);
	Mat bin5 = Mat(src.rows, src.cols, CV_8U);
	Mat bin6 = Mat(src.rows, src.cols, CV_8U);
	Mat bin7 = Mat(src.rows, src.cols, CV_8U);
	Mat bin8 = Mat(src.rows, src.cols, CV_8U);
	Mat bin9 = Mat(src.rows, src.cols, CV_8U);
	Mat bin10 = Mat(src.rows, src.cols, CV_8U);
	Mat bin11 = Mat(src.rows, src.cols, CV_8U);
	Mat bin12 = Mat(src.rows, src.cols, CV_8U);
	Mat bin13 = Mat(src.rows, src.cols, CV_8U);
	Mat bin14 = Mat(src.rows, src.cols, CV_8U);
	Mat bin15 = Mat(src.rows, src.cols, CV_8U);
	
	gettimeofday(&tv, NULL);

	uint64 start = tv.tv_usec;
	
	// Convert from micro seconds (10^-6) to milliseconds (10^-3)
	//start /= 1000;

	// Adds the seconds (10^0) after converting them to milliseconds (10^-3)
	//start += (tv.tv_sec * 1000);
	
	histograms_of_gradient_directions_sse(src.data, bin0.data, bin1.data, bin2.data, bin3.data, bin4.data, bin5.data, bin6.data, bin7.data, bin8.data, bin9.data, bin10.data, bin11.data, bin12.data, bin13.data, bin14.data, bin15.data, src.cols, src.rows, 5);

	gettimeofday(&tv, NULL);

	uint64 end = tv.tv_usec;
	
	// Convert from micro seconds (10^-6) to milliseconds (10^-3)
	//end /= 1000;

	// Adds the seconds (10^0) after converting them to milliseconds (10^-3)
	//end += (tv.tv_sec * 1000);

	cout << "Time: " << (end - start) << endl;

	imshow("hog",bin0);
	waitKey();
	imshow("hog",bin1);
	waitKey();
	imshow("hog",bin2);
	waitKey();
	imshow("hog",bin3);
	waitKey();
	imshow("hog",bin4);
	waitKey();
	imshow("hog",bin5);
	waitKey();
	imshow("hog",bin6);
	waitKey();
	imshow("hog",bin7);
	waitKey();
	imshow("hog",bin8);
	waitKey();
	imshow("hog",bin9);
	waitKey();
	imshow("hog",bin10);
	waitKey();
	imshow("hog",bin11);
	waitKey();
	imshow("hog",bin12);
	waitKey();
	imshow("hog",bin13);
	waitKey();
	imshow("hog",bin14);
	waitKey();
	imshow("hog",bin15);
	waitKey();

	return 0;
}

