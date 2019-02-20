#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Inpainting.hpp"

using namespace cv;
using namespace std;

extern "C" {
void patchMatch (char* color, char* mask, char* output) {
  string texture_file_name = "";
  string color_file_name = color;
  string mask_file_name = mask;
  string output_file_name = output;
  cout << color_file_name << mask_file_name << output_file_name;
  Inpainting ip(color_file_name, mask_file_name, texture_file_name);
  Mat inpainted_img = ip.inpaint();
  imwrite(output_file_name, inpainted_img);
}
}
