#include "Inpainting.hpp"

#define RADIUS 4
#define BORDER_RADIUS 4

Inpainting::Inpainting(string color_file_name,
    string mask_file_name,
    string texture_file_name) {
  std::cout << color_file_name << mask_file_name;
  // Check inputs
  assert(color_file_name.length() && mask_file_name.length());

  color_mat_ = imread(color_file_name, CV_LOAD_IMAGE_COLOR);
  mask_mat_ = imread(mask_file_name, CV_LOAD_IMAGE_GRAYSCALE);
  texture_mat_ = imread(texture_file_name, CV_LOAD_IMAGE_COLOR);

  assert(color_mat_.size() == mask_mat_.size());
std:: cout << color_mat_.size() << mask_mat_.size();
std:: cout << color_mat_.empty() << mask_mat_.empty();
  assert(!color_mat_.empty() && !mask_mat_.empty());

  // Convert color_mat_ to depth CV_32F for colorspace conversions
  color_mat_.convertTo(color_mat_, CV_32FC3);
  color_mat_ /= 255.0f;

  // Add border around color_mat_
  copyMakeBorder(
      color_mat_, color_mat_,
      RADIUS, RADIUS, RADIUS, RADIUS,
      BORDER_CONSTANT, Scalar_<float>(0,0,0));

  cvtColor(color_mat_, gray_mat_, CV_BGR2GRAY);

  // Preare texture_mat_
  if (texture_mat_.data) {
    texture_mat_.convertTo(texture_mat_, CV_32FC3);
    texture_mat_ /= 255.0f;

    copyMakeBorder(
        texture_mat_, texture_mat_,
        RADIUS, RADIUS, RADIUS, RADIUS,
        BORDER_CONSTANT, Scalar_<float>(0,0,0));
  } else {
    texture_mat_ = color_mat_;
  }
}

int Inpainting::mod(int a, int b) {
  return ((a % b) + b) % b;
}

void Inpainting::getContours(const Mat& mask,
    contours_t& contours, hierarchy_t& hierarchy) {
  findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
}

Point2f Inpainting::getNormal(const contour_t& contour, const Point& point) {
  int sz = (int) contour.size();

  assert(sz != 0);

  int point_index = (int) (find(contour.begin(), contour.end(), point) - contour.begin());

  assert(point_index != contour.size());

  if (sz == 1) {
    return Point2f(1.0f, 0.0f);
  } else if (sz < 2 * BORDER_RADIUS + 1) {
    // Too few points in contour to use LSTSQ regression
    // return the normal with respect to adjacent neigbourhood
    Point adj = contour[(point_index + 1) % sz] - contour[point_index];
    return Point2f(adj.y, -adj.x) / norm(adj);
  }

  // Use least square regression
  // create X and Y mat to SVD
  Mat X(Size(2, 2 * BORDER_RADIUS+1), CV_32F);
  Mat Y(Size(1, 2 * BORDER_RADIUS+1), CV_32F);

  assert(X.rows == Y.rows && X.cols == 2 && Y.cols == 1
      && X.type() == Y.type()
      && Y.type() == CV_32F);

  int i = mod((point_index - BORDER_RADIUS), sz);

  float* X_row;
  float* Y_row;

  int count = 0;
  int count_X_equal = 0;
  while (count < 2 * BORDER_RADIUS+1)
  {
    X_row = X.ptr<float>(count);
    X_row[0] = contour[i].x;
    X_row[1] = 1.0f;

    Y_row = Y.ptr<float>(count);
    Y_row[0] = contour[i].y;

    if (X_row[0] == contour[point_index].x)
    {
      ++count_X_equal;
    }

    i = mod(i+1, sz);
    ++count;
  }

  if (count_X_equal == count) {
    return Point2f(1.0f, 0.0f);
  }
  // to find the line of best fit
  Mat sol;
  solve(X, Y, sol, DECOMP_SVD);

  assert(sol.type() == CV_32F);

  float slope = sol.ptr<float>(0)[0];
  Point2f normal(-slope, 1);

  return normal / norm(normal);
}

Mat Inpainting::getPatch(const Mat& mat, const Point& p) {
  return mat(
      Range(p.y - RADIUS, p.y + RADIUS+1),
      Range(p.x - RADIUS, p.x + RADIUS+1)
      );
}

void Inpainting::computePriority(const contours_t& contours, const Mat& gray_mat,
    const Mat& confidence_mat, Mat& priority_mat, Point last_point) {
  // get the derivatives and magnitude of the greyscale image
  Mat dx, dy, magn;
  Sobel(gray_mat, dx, -1, 1, 0, -1);
  Sobel(gray_mat, dy, -1, 0, 1, -1);
  magnitude(dx, dy, magn);

  // mask the magnitude
  Mat masked_magnitude(magn.size(), magn.type(), Scalar_<float>(0));
  magn.copyTo(masked_magnitude, (confidence_mat != 0.0f));
  erode(masked_magnitude, masked_magnitude, Mat());

  bool is_update = true;
  if (last_point.x == -1 && last_point.y == -1)
    is_update = false;
#pragma omp parallel
  {
#pragma omp for schedule(dynamic)
    for (int i = 0; i < contours.size(); ++i)
    {
      contour_t contour = contours[i];

      for (int j = 0; j < contour.size(); ++j)
      {

        Point point = contour[j];
        if (!is_update ||
            (abs(point.x - last_point.x) < 2 * RADIUS &&
            abs(point.y - last_point.y) < 2 * RADIUS)) {

          Mat confidence_patch = getPatch(confidence_mat, point);

          // get confidence of patch
          double confidence = sum(confidence_patch)[0] / (double) confidence_patch.total();
          assert(0 <= confidence && confidence <= 1.0f);

          // get the normal to the border around point
          Point2f normal = getNormal(contour, point);

          // get the maximum gradient in source around patch
          Mat magnitude_patch = getPatch(masked_magnitude, point);
          Point max_point;
          minMaxLoc(magnitude_patch, NULL, NULL, NULL, &max_point);
          Point2f gradient = Point2f(
              -getPatch(dy, point).ptr<float>(max_point.y)[max_point.x],
              getPatch(dx, point).ptr<float>(max_point.y)[max_point.x]
              );

          // set the priority in priorityMat
          priority_mat.ptr<float>(point.y)[point.x] = abs((float) confidence * gradient.dot(normal));
          assert(priority_mat.ptr<float>(point.y)[point.x] >= 0);
        }
      }
    }
  }
}

Mat Inpainting::computeSSD(const Mat& tmplate, const Mat& source, const Mat& tmplate_mask) {
  Mat result(source.rows - tmplate.rows + 1, source.cols - tmplate.cols + 1, CV_32F, 0.0f);

  //matchTemplate(source, tmplate, result, CV_TM_SQDIFF, tmplate_mask);
  gpu_matchTemplate(source, tmplate, result, CV_TM_SQDIFF, tmplate_mask);

  normalize(result, result, 0, 1, NORM_MINMAX);
  copyMakeBorder(result, result,
      RADIUS, RADIUS, RADIUS, RADIUS,
      BORDER_CONSTANT, 1.1f);
  return result;
}

void Inpainting::transferPatch(const Point& psiHatQ, const Point& psiHatP, const Mat& src_mat, Mat& dst_mat, const Mat& mask_mat) {
  // copy contents of psiHatQ to psiHatP with mask
  getPatch(src_mat, psiHatQ).copyTo(getPatch(dst_mat, psiHatP), getPatch(mask_mat, psiHatP));
}

double Inpainting::computeConfidence(const Mat& confidence_patch) {
  return sum(confidence_patch)[0] / (double) confidence_patch.total();
}

Mat Inpainting::inpaint() {
  // Prepare mask_mat_ and confidence_mat_
  Mat confidence_mat;
  mask_mat_.convertTo(confidence_mat, CV_32F);
  confidence_mat /= 255.0f;
  copyMakeBorder(
      mask_mat_, mask_mat_,
      RADIUS, RADIUS, RADIUS, RADIUS,
      BORDER_CONSTANT, 255);
  copyMakeBorder(
      confidence_mat, confidence_mat,
      RADIUS, RADIUS, RADIUS, RADIUS,
      BORDER_CONSTANT, 0.0001f);

  Mat priority_mat(confidence_mat.size(), CV_32FC1);

  contours_t contours;
  hierarchy_t hierarchy;

  Point psiHatP(-1, -1); // psiHatP - point of highest confidence
  Mat psiHatPColor; // color patch around psiHatP
  Mat psiHatPConfidence; // confidence patch around psiHatP

  Mat template_mask; // mask for template match (3 channel)
  Mat result; // holds result from template matching

  Mat eroded_mask; // eroded mask

  Point psiHatQ; // psiHatQ - point of closest patch
  double confidence; // confidence of psiHatPConfidence

  // eroded mask is used to ensure that psiHatQ is not overlapping with target
  erode(mask_mat_, eroded_mask, Mat(), Point(-1, -1), RADIUS);

  Mat draw_mat;

  // main loop
  const size_t area = mask_mat_.total();
  priority_mat.setTo(-0.1f);
  while (countNonZero(mask_mat_) != area) // end when target is filled
  {
    // get the contours of mask
    getContours((mask_mat_ == 0), contours, hierarchy);
    Mat img_contour = Mat::zeros(color_mat_.size(), CV_8UC3);
    int contour_idx = -1;
    Scalar color(0, 0, 255);
    drawContours(img_contour, contours, contour_idx, color);
    imshow("Bg2:, contour", img_contour);

    // compute the priority for all contour points
    computePriority(contours, gray_mat_, confidence_mat, priority_mat, psiHatP);

    // get the patch with the greatest priority
    minMaxLoc(priority_mat, NULL, NULL, NULL, &psiHatP);
    psiHatPColor = getPatch(color_mat_, psiHatP);
    psiHatPConfidence = getPatch(confidence_mat, psiHatP);

    // update priority_mat
    getPatch(priority_mat, psiHatP) = -0.1f;

    Mat conf_inv = (psiHatPConfidence != 0.0f);
    conf_inv.convertTo(conf_inv, CV_32F);
    conf_inv /= 255.0f;
    // get the patch in source with least distance to psiHatPColor wrt source of psiHatP
    Mat merge_arrays[3] = {conf_inv, conf_inv, conf_inv};
    merge(merge_arrays, 3, template_mask);
    result = computeSSD(psiHatPColor, texture_mat_, template_mask);

    // set all target regions to 1.1, which is over the maximum value possilbe
    // from SSD
    result.setTo(1.1f, eroded_mask == 0);
    // get minimum point of SSD between psiHatPColor and colorMat
    cuda::minMaxLoc(result, NULL, NULL, &psiHatQ, NULL);

    assert(psiHatQ != psiHatP);

    // updates
    // copy from psiHatQ to psiHatP for each colorspace
    transferPatch(psiHatQ, psiHatP, gray_mat_, gray_mat_, (mask_mat_ == 0));
    transferPatch(psiHatQ, psiHatP, texture_mat_, color_mat_, (mask_mat_ == 0));

    // fill in confidenceMat with confidences C(pixel) = C(psiHatP)
    confidence = computeConfidence(psiHatPConfidence);
    assert(0 <= confidence && confidence <= 1.0f);
    // update confidence
    psiHatPConfidence.setTo(confidence, (psiHatPConfidence == 0.0f));
    // update maskMat
    mask_mat_ = (confidence_mat != 0.0f);
    //imshow("Bg1", color_mat_);
    //waitKey(1);
    cout << area - countNonZero(mask_mat_) << " remaining pixels.\r" << flush;
  }

  Mat inpainted_img = color_mat_(Range(RADIUS, color_mat_.rows - RADIUS), Range(RADIUS, color_mat_.cols - RADIUS));
  inpainted_img *= 255;
  inpainted_img.convertTo(inpainted_img, CV_8UC3);
  return inpainted_img;
}
