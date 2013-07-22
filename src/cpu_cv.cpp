#include "cpu_cv.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

const int openCVtest()
{
    static const char* WINDOW_NAME = "Webcam Test";

    CvCapture* capture = nullptr;
    Mat frame, frameCopy;

    capture = cvCaptureFromCAM(CV_CAP_ANY);
    if (capture == nullptr)
    {
        cout << "No camera detected" << endl;
        return 1;
    }

    cvNamedWindow(WINDOW_NAME, CV_WINDOW_AUTOSIZE);

    cout << "In capture ..." << endl;
    for(;;)
    {
        IplImage* iplImg = cvQueryFrame(capture);
        if (iplImg == nullptr)
            continue;

        frame = iplImg;
        if(frame.empty())
            continue;

        frame = Sobel(frame, true);
        //frame = HarrisCorners(frame);

        imshow(WINDOW_NAME, frame);

        if(waitKey(10) >= 0)
            break;
    }

    cvReleaseCapture(&capture); // capture cannot be nullptr because we return if == nullptr earlier
    cvDestroyWindow(WINDOW_NAME);

    return 0;
}

Mat HarrisCorners(Mat& src)
{
    // Detector parameters
    static const int blockSize = 2, apertureSize = 3, thresh = 150;
    static const double k = 0.04;

    static Mat dst, dst_norm_scaled;
    dst = Mat::zeros(src.size(), CV_32FC1);

    // Convert it to gray
    cvtColor(src, src, CV_RGB2GRAY);

    // Detecting corners
    cornerHarris(src, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    // Normalizing
    normalize(dst, dst, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    // convertScaleAbs(dst, dst_norm_scaled);

    // Drawing a circle around corners
    for (int j=0; j<dst.rows; ++j)
    {
        for (int i=0; i<dst.cols; ++i)
        {
            if((int)dst.at<float>(j,i) > thresh) // so really a corner is a value that exceeds some threshold in dst
            {
                circle(src, Point(i, j), 5,  Scalar(255, 0, 0), 2, 8, 0);
            }
        }
    }

    return src;
}

Mat Sobel(Mat& src, const bool do_blur=false)
{
    const int scale = 1, delta = 0, ddepth = CV_16S;

    static Mat grad_x, grad_y;

    if (do_blur)
        GaussianBlur(src, src, Size(3,3), 0, 0, BORDER_DEFAULT);

    // Convert it to gray
    cvtColor(src, src, CV_RGB2GRAY);

    // Gradient X
    Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, grad_x);

    // Gradient Y
    Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, grad_y);

    // Total Gradient (approximate 0.5*|g_x|+0.5*|g_y|)
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, src);

    return src;
}
