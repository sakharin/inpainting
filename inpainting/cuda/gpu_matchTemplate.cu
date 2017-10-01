#include "../gpu_matchTemplate.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;

struct st_matchTemplate {
  PtrStepSzf img_;
  PtrStepSzf tem_;
  mutable PtrStepSzf res_;

  int img_w_, img_h_;
  int tem_w_, tem_h_;
  int res_w_, res_h_;

  bool have_mask_;
  PtrStepSzf mas_;

  __device__ __forceinline__ float cal_cost(float img, float tmp) const {
    float diff = img - tmp;
    return diff * diff;
  }

  __device__ __forceinline__ void matchTemplate() const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= res_w_ || y >= res_h_)
      return;

    float cost = 0;
    for (int j = 0; j < tem_h_; j++) {
      for (int i = 0; i < tem_w_; i++) {
        int u = 3 * i;
		int v = j;
        int xi = 3 * x + u;
        int yj = y + v;

        float c = 0;
		if (have_mask_) {
          c += cal_cost(img_.ptr(yj)[xi    ], tem_.ptr(v)[u    ]) * mas_.ptr(v)[u    ];
          c += cal_cost(img_.ptr(yj)[xi + 1], tem_.ptr(v)[u + 1]) * mas_.ptr(v)[u + 1];
          c += cal_cost(img_.ptr(yj)[xi + 2], tem_.ptr(v)[u + 2]) * mas_.ptr(v)[u + 2];
		} else {
          c += cal_cost(img_.ptr(yj)[xi    ], tem_.ptr(v)[u    ]);
          c += cal_cost(img_.ptr(yj)[xi + 1], tem_.ptr(v)[u + 1]);
          c += cal_cost(img_.ptr(yj)[xi + 2], tem_.ptr(v)[u + 2]);
		}
        cost += c;
      }
    }
    res_.ptr(y)[x] = cost;
  }
};

__global__ void kn_matchTemplate(const st_matchTemplate st) {
  st.matchTemplate();
}

void gpu_matchTemplate(Mat image, Mat templ, Mat& result, int method, Mat mask) {
  // Check input
  if (!image.data || !templ.data ||
      image.rows <= templ.rows ||
      image.cols <= templ.cols ||
      image.type() != CV_32FC3 ||
      templ.type() != CV_32FC3) {
    cout << "Error! Image and template must be an 32-bit floating-point RGB image and template";
    cout << " must not be bigger than image." << endl;
    exit(-1);
  }

  st_matchTemplate st;

  st.img_w_  = image.cols;
  st.img_h_ = image.rows;
  st.tem_w_ = templ.cols;
  st.tem_h_ = templ.cols;
  st.res_w_ = st.img_w_ - st.tem_w_ + 1;
  st.res_h_ = st.img_h_ - st.tem_h_ + 1;

  result.create(st.res_h_, st.res_w_, CV_32FC1);
  GpuMat d_img, d_tem, d_res;
  d_img.upload(image);
  d_tem.upload(templ);
  d_res.upload(result);
  st.img_ = d_img;
  st.tem_ = d_tem;
  st.res_ = d_res;

  if (!(method == CV_TM_SQDIFF)) {
    cout << "Support method is SQDIFF." << endl;
    exit(-1);
  }

  st.have_mask_ = false;
  if (!mask.data ||
      mask.cols != templ.cols || mask.rows != templ.rows ||
      mask.type() != CV_32FC3) {
    cout << "Mask image must be an 32-bit floating-point rgb image";
    cout << " and it must have the same size as template." << endl;
    exit(-1);
  } else {
    GpuMat d_mas;
    d_mas.upload(mask);
    st.mas_ = d_mas;
	st.have_mask_ = true;
  }

  const dim3 block(32, 8);
  const dim3 grid(divUp(st.res_w_, block.x), divUp(st.res_h_, block.y));

  kn_matchTemplate<<<grid, block>>>(st);
  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());

  d_res.download(result);
}
