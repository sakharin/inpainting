#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Inpainting.hpp"

using namespace cv;
using namespace std;

int main (int argc, char** argv) {
  string color_file_name, mask_file_name, texture_file_name, output_file_name;

  if (argc == 4) {
    color_file_name = argv[1];
    mask_file_name = argv[2];
    output_file_name = argv[3];
    texture_file_name = "";
  } else if (argc == 5) {
    color_file_name = argv[1];
    mask_file_name = argv[2];
    texture_file_name = argv[3];
    output_file_name = argv[4];
  } else {
    cerr << "Usage: ./inpainting colorImageFile maskImageFile outputImageFile" << endl;
    cerr << "Usage: ./inpainting colorImageFile maskImageFile textureFile outputImageFile" << endl;
    return -1;
  }

  Inpainting ip(color_file_name, mask_file_name, texture_file_name);
  Mat inpainted_img = ip.inpaint();
  imwrite(output_file_name, inpainted_img);
  return 0;
}
