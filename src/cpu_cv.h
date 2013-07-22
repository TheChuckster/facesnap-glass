#ifndef CPU_CV_H_
#define CPU_CV_H_

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat HarrisCorners(cv::Mat& src);
cv::Mat Sobel(cv::Mat& src, const bool do_blur);

const int openCVtest();

#endif
