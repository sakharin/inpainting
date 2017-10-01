#ifndef __GPU_MATCHTEMPLATE_H__
#define __GPU_MATCHTEMPLATE_H__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

void gpu_matchTemplate(Mat image, Mat templ, Mat& result, int method, Mat mask=Mat());
#endif
