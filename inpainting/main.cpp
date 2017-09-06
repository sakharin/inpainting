//
//  main.cpp
//  An example main function showcasing how to use the inpainting function.
//
//  Created by Sooham Rafiz on 2016-05-16.

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <string>

#include "utils.hpp"

using namespace cv;
using namespace std;

/*
 * Note: This program uses C assert() statements, define NDEBUG marco to
 * disable assertions.
 */

#ifndef DEBUG
   #define DEBUG 0
#endif

int main (int argc, char** argv) {
  // --------------- read filename strings ------------------
  string colorFilename, maskFilename, outputFilename;

  if (argc == 4) {
    colorFilename = argv[1];
    maskFilename = argv[2];
    outputFilename = argv[3];
  } else {
    cerr << "Usage: ./inpainting colorImageFile maskImageFile outputImageFile" << endl;
    return -1;
  }

  // ---------------- read the images ------------------------
  // colorMat - color picture + border
  // maskMat  - mask picture + border
  // grayMat  - gray picture + border
  Mat colorMat, maskMat, grayMat;
  loadInpaintingImages(
      colorFilename,
      maskFilename,
      colorMat,
      maskMat,
      grayMat
      );

  // confidenceMat - confidence picture + border
  Mat confidenceMat;
  maskMat.convertTo(confidenceMat, CV_32F);
  confidenceMat /= 255.0f;

  // add borders around maskMat and confidenceMat
  copyMakeBorder(
      maskMat, maskMat,
      RADIUS, RADIUS, RADIUS, RADIUS,
      BORDER_CONSTANT, 255);
  copyMakeBorder(
      confidenceMat, confidenceMat,
      RADIUS, RADIUS, RADIUS, RADIUS,
      BORDER_CONSTANT, 0.0001f);

  // ---------------- start the algorithm -----------------

  contours_t contours;   // mask contours
  hierarchy_t hierarchy; // contours hierarchy


  // priorityMat - priority values for all contour points + border
  Mat priorityMat(
      confidenceMat.size(),
      CV_32FC1
      ); // priority value matrix for each contour point

  assert(
      colorMat.size() == grayMat.size() &&
      colorMat.size() == confidenceMat.size() &&
      colorMat.size() == maskMat.size()
      );

  Point psiHatP; // psiHatP - point of highest confidence

  Mat psiHatPColor; // color patch around psiHatP

  Mat psiHatPConfidence; // confidence patch around psiHatP
  double confidence; // confidence of psiHatPConfidence

  Point psiHatQ; // psiHatQ - point of closest patch

  Mat result; // holds result from template matching
  Mat erodedMask; // eroded mask

  Mat templateMask; // mask for template match (3 channel)

  // eroded mask is used to ensure that psiHatQ is not overlapping with target
  erode(maskMat, erodedMask, Mat(), Point(-1, -1), RADIUS);

  Mat drawMat;


  // main loop
  const size_t area = maskMat.total();

  while (countNonZero(maskMat) != area) // end when target is filled
  {
    // set priority matrix to -.1, lower than 0 so that border area is never selected
    priorityMat.setTo(-0.1f);

    // get the contours of mask
    getContours((maskMat == 0), contours, hierarchy);
    Mat img_contour = Mat::zeros(colorMat.size(), CV_8UC3);
    int contourIdx = -1;
    Scalar color(0, 0, 255);
    drawContours(img_contour, contours, contourIdx, color);
    imshow("Bg2:, contour", img_contour);

    if (DEBUG) {
      drawMat = colorMat.clone();
    }

    // compute the priority for all contour points
    computePriority(contours, grayMat, confidenceMat, priorityMat);

    // get the patch with the greatest priority
    minMaxLoc(priorityMat, NULL, NULL, NULL, &psiHatP);
    psiHatPColor = getPatch(colorMat, psiHatP);
    psiHatPConfidence = getPatch(confidenceMat, psiHatP);

    Mat confInv = (psiHatPConfidence != 0.0f);
    confInv.convertTo(confInv, CV_32F);
    confInv /= 255.0f;
    // get the patch in source with least distance to psiHatPColor wrt source of psiHatP
    Mat mergeArrays[3] = {confInv, confInv, confInv};
    merge(mergeArrays, 3, templateMask);
    result = computeSSD(psiHatPColor, colorMat, templateMask);

    // set all target regions to 1.1, which is over the maximum value possilbe
    // from SSD
    result.setTo(1.1f, erodedMask == 0);
    // get minimum point of SSD between psiHatPColor and colorMat
    minMaxLoc(result, NULL, NULL, &psiHatQ);

    assert(psiHatQ != psiHatP);

    if (DEBUG) {
      rectangle(drawMat, psiHatP - Point(RADIUS, RADIUS), psiHatP + Point(RADIUS+1, RADIUS+1), Scalar(255, 0, 0));
      rectangle(drawMat, psiHatQ - Point(RADIUS, RADIUS), psiHatQ + Point(RADIUS+1, RADIUS+1), Scalar(0, 0, 255));
      showMat("red - psiHatQ", drawMat);
    }
    // updates
    // copy from psiHatQ to psiHatP for each colorspace
    transferPatch(psiHatQ, psiHatP, grayMat, (maskMat == 0));
    transferPatch(psiHatQ, psiHatP, colorMat, (maskMat == 0));

    // fill in confidenceMat with confidences C(pixel) = C(psiHatP)
    confidence = computeConfidence(psiHatPConfidence);
    assert(0 <= confidence && confidence <= 1.0f);
    // update confidence
    psiHatPConfidence.setTo(confidence, (psiHatPConfidence == 0.0f));
    // update maskMat
    maskMat = (confidenceMat != 0.0f);
    imshow("Bg1", colorMat);
    waitKey(1);
    cout << colorMat.rows * colorMat.cols - countNonZero(maskMat) << " remaining pixels.\r" << flush;
  }

  cout << endl << "done" << endl;
  showMat("final result", colorMat, 1);
  Mat out = colorMat.clone() * 255;
  out.convertTo(out, CV_8UC3);
  imwrite(outputFilename, out);
  return 0;
}
