#include <iostream>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace cv;
using namespace cv::cuda;

const int X_CORDINATE = 575;
const int Y_CORDINATE = 45;
const int WIDTH =  60;
const int HEIGHT = 50;

const int CONST_AREA_THRESHOLD = 30;
const int CONST_FLAME_THRESHOLD = 10;
const int THICKNESS = 2;

bool Flame_Area_Detection(const Mat &src){
    // Converting grayscale to black n white
    cv::Mat bw = src > .5;
    cv::blur( bw, bw, cv::Size(3,3) );
   
    int count_white = 0;
    for(int i =0 ; i< bw.rows; i++){
        for(int j =0 ; j< bw.cols; j++){
            if(bw.at<float>(i,j) != 0){
              count_white++;
            }
        }
    }

    float per_b_pixel =((float)count_white/(float)(bw.rows*bw.cols))* 100;
    if(per_b_pixel < CONST_AREA_THRESHOLD)
        return false;
    return true;
}

bool get_HSI(const Mat &pic){

    cv::Mat hsi(pic.rows, pic.cols, pic.type());
    int H = pic.rows;
    int W = pic.cols;
    bool grab = true;
    for (int j=0;j<H;j++)
    for (int i=0;i<W;i++) {

        double B =(double) pic.at<cv::Vec3b>(j,i)[0];
        double G =(double) pic.at<cv::Vec3b>(j,i)[1];
        double R =(double) pic.at<cv::Vec3b>(j,i)[2];
        double intensity = 0;
        double hue = 0;
        double saturation = 0;
    
        intensity = (double) (R + G + B) / (3.0);
        double tmp = min(R, min(G, B));

        saturation = 1 - 3*(tmp/(R + G + B));
        if(saturation < 0.00001){
            saturation = 0;
        }else if(saturation > 0.99999){
                saturation = 1;
        }

        double rb = R - B;
		double rg = R - G;
		double denom = sqrt(( rg * rg) + (rb*(G - B)));
		if( saturation != 0 and denom!=0){
            hue = 0.5 * (rg - rb) /sqrt(( rg * rg) + (rb*(G - B)));
			hue = acos(hue);
			if( B <= G){
                    hue = hue;
            }
			else{
                    hue = 360 -hue;
            }
        }		
		else{
                    hue = 180;
        }
        saturation = saturation * 100;
        hsi.at<cv::Vec3b>(j, i)[2] = hue;
        hsi.at<cv::Vec3b>(j, i)[1] = saturation;
        hsi.at<cv::Vec3b>(j, i)[0] = intensity;
        }

    int count = 0;
	for (int j=0;j< H;j++)
       for (int i=0;i<W;i++){

        double I =(double) hsi.at<cv::Vec3b>(j,i)[0];
        double S =(double) hsi.at<cv::Vec3b>(j,i)[1];
        double H =(double) hsi.at<cv::Vec3b>(j,i)[2];
		//Tune values of H,S,I as per demand 
		if (!((H<=120) and (S<=50) and (I>=180))){
                hsi.at<cv::Vec3b>(j, i)[2] = 0;
                hsi.at<cv::Vec3b>(j, i)[1] = 0;
                hsi.at<cv::Vec3b>(j, i)[0] = 0;
				count+=1;
        }
    }

	int size = (H * W);
	float flame_per = 100 - (count*100)/size;

	if (flame_per < CONST_FLAME_THRESHOLD){
              return false;
    }

    //cout<< flame_per<<endl;
    cv::imshow("HSI Format",hsi);
    return true;
}


bool Motion_Detection(const Mat &prev ,const  Mat &next) {
    Mat flow;
    cv::calcOpticalFlowFarneback(prev,next,flow, 0.5,3,15,3,5,1.2,0);
    cv::Mat xy[2]; //X,Y
    cv::split(flow, xy);

    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

    double mag_max;
    cv::minMaxLoc(magnitude, 0, &mag_max);
    Mat mag;
    
    normalize(magnitude, mag, 0, 255, NORM_MINMAX);
    cv::Mat _hsv[3], hsv;
    _hsv[0] = angle ; //((angle*180)/3.14)/2;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[1] = 255;
    _hsv[2] = mag;
    cv::merge(_hsv, 3, hsv);

    //convert to BGR and show
    cv::Mat bgr;//CV_32FC3 matrix
    cuda::GpuMat hsv_cuda, bgr_cuda;
    hsv_cuda.upload(hsv);
    cuda::cvtColor(hsv_cuda, bgr_cuda, cv::COLOR_HSV2BGR);
    bgr_cuda.download(bgr);

    int count = 0;
    for(int i =0 ; i<bgr.size[0]; i++ ){
        for(int j =0 ; j<bgr.size[1]; j++ ){
            if(bgr.at<float>(i,j) != 0){
                    //cout<< bgr.at<Vec3b>(i,j)[0];
                    count++;
            }
        }
    }   
     int thresh_hold = (bgr.size[0] * bgr.size[1])/3;
     if(count >= thresh_hold){
         cv::imshow("optical flow", bgr);
         return (true) ;
     }
    cv::imshow("optical flow", bgr);
    return false;
}

int main() {

    //Reading videa frames from the dir.
    VideoCapture cap("./../Data/flame.mkv");
    if (!cap.isOpened())
        return -1;

    // Declaring Mat and GPU variables
    GpuMat gpu_frame;
    Mat frame;
    cv::Mat greyMat;
    cv::Mat prevFrame;
    GpuMat prev_frame;
    bool grabFrame = true;
    bool r1,r2,r3;

    cap >> frame;

    frame.convertTo(frame, CV_32F);
    gpu_frame.upload(frame);
    gpu_frame.convertTo(gpu_frame,CV_32F, 1.f/ 255);

    cv::cuda::GpuMat image(gpu_frame);
    cv::Rect regionOfInterest( X_CORDINATE ,Y_CORDINATE, WIDTH, HEIGHT);
    gpu_frame = image(regionOfInterest);

    cv::cuda::cvtColor(gpu_frame, prev_frame, CV_BGR2GRAY);
    prev_frame.download(prevFrame);

    while (grabFrame) {
        // Grab frame
        cap >> frame;
        Mat frame_HSI = frame;
        frame.convertTo(frame, CV_32F);
        gpu_frame.upload(frame);

        gpu_frame.convertTo(gpu_frame,CV_32F, 1.f/ 255);
        cv::cuda::GpuMat image(gpu_frame);
        cv::Rect regionOfInterest( X_CORDINATE ,Y_CORDINATE, WIDTH, HEIGHT);
        gpu_frame = image(regionOfInterest);

        GpuMat next_frame;
        cv::cuda::cvtColor(gpu_frame, next_frame, CV_BGR2GRAY);
        next_frame.download(greyMat);
        Mat croppedImg;
        frame_HSI(cv::Rect(X_CORDINATE,Y_CORDINATE,WIDTH,HEIGHT)).copyTo(croppedImg);

        r1 = Motion_Detection( prevFrame , greyMat);
        r2 = get_HSI(croppedImg);
        r3 = Flame_Area_Detection(greyMat);

        //r1 && r2 && r3
        if(r1 && r2 && r3){
            cv::rectangle(frame_HSI ,regionOfInterest, cv::Scalar(0, 0, 255),THICKNESS);
        }else{
            	cout<<"###############################################"<<endl;
		        cout<<"Is Frame showing movement :::::::::::::::: "<<r1<<endl;
		        cout<<"Is Flame content valid ::::::::::::::::::: "<<r2<<endl;
		        cout<<"Is Flame size valid :::::::::::::::::::::: "<<r3<<endl;
		        cout <<"###############################################"<<endl;
            cv::rectangle(frame_HSI ,regionOfInterest, cv::Scalar(0, 255, 0), THICKNESS);
        }

        prev_frame = next_frame;
        next_frame.download(prevFrame);

        if (frame.empty())
            {
                cout<<"Video Ended Successfully"<<endl;
                break;
            }
        imshow("Flame Detection", frame_HSI);
        if (cv::waitKey(30) >= 0)
            grabFrame = false;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

