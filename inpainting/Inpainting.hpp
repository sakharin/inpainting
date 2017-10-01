#ifndef __INPAINTING_H__
#define __INPAINTING_H__

#include "gpu_matchTemplate.hpp"
#include "omp.h"

typedef vector< vector< Point > > contours_t;
typedef vector< Vec4i > hierarchy_t;
typedef vector< Point > contour_t;

class Inpainting {
 private:
  Mat color_mat_;
  Mat mask_mat_;
  Mat gray_mat_;
  Mat texture_mat_;

  int mod(int a, int b);
  void getContours(const Mat& mask,
      contours_t& contours, hierarchy_t& hierarchy);
  Point2f getNormal(const contour_t& contour, const Point& point);
  Mat getPatch(const Mat& mat, const Point& p);
  void computePriority(const contours_t& contours,
      const Mat& gray_mat, const Mat& confidence_mat, Mat& priority_mat);
  Mat computeSSD(const Mat& tmplate, const Mat& source, const Mat& tmplate_mask);
  void transferPatch(const Point& psiHatQ, const Point& psiHatP, const Mat& src_mat, Mat& dst_mat, const Mat& mask_mat);
  double computeConfidence(const Mat& confidence_patch);
 public:
  Inpainting(string color_file_name,
      string mask_file_name,
      string texture_file_name="");
  Mat inpaint();
};
#endif
