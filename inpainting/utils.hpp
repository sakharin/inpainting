#ifndef utils_hpp
#define utils_hpp

#include <iostream>
#include <assert.h>

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include "omp.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

typedef vector< vector< Point > > contours_t;
typedef vector< Vec4i > hierarchy_t;
typedef vector< Point > contour_t;


// Patch raduius
#define RADIUS 4
// The maximum number of pixels around a specified point on the target outline
#define BORDER_RADIUS 4

int mod(int a, int b);

void loadInpaintingImages(
    const string& colorFilename,
    const string& maskFilename,
    Mat& colorMat,
    Mat& maskMat,
    Mat& grayMat);

void showMat(const String& winname, const Mat& mat, int time=5);

void getContours(const Mat& mask, contours_t& contours, hierarchy_t& hierarchy);

double computeConfidence(const Mat& confidencePatch);

Mat getPatch(const Mat& image, const Point& p);

void getDerivatives(const Mat& grayMat, Mat& dx, Mat& dy);

Point2f getNormal(const contour_t& contour, const Point& point);

void computePriority(const contours_t& contours, const Mat& grayMat, const Mat& confidenceMat, Mat& priorityMat);

void transferPatch(const Point& psiHatQ, const Point& psiHatP, Mat& mat, const Mat& maskMat);

Mat computeSSD(const Mat& tmplate, const Mat& source, const Mat& tmplateMask);

#endif
