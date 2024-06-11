#pragma once

#include <torch/torch.h>

#include <DiffusionModelC++/Util/Logging.hpp>
#include <opencv2/opencv.hpp>

namespace dmcpp {
namespace util {

inline cv::Mat loadImage(const std::string& filePath) {
  cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

  if (image.empty()) {
    LOG_CRITICAL("Could not open or find the image: " + filePath);
    exit(EXIT_FAILURE);
  }

  return image;
}

inline torch::Tensor cv2MatToTensor(const cv::Mat& image, bool normalize = true) {
  cv::Mat floatImage;
  image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

  // NOTE: BGR -> RGB
  cv::cvtColor(floatImage, floatImage, cv::COLOR_BGR2RGB);

  // NOTE: HWC (Height, Width, Channels) -> CHW (Channels, Height, Width)
  cv::Mat chw_image = floatImage.reshape(1, {floatImage.rows, floatImage.cols, 3});
  torch::Tensor chw_image_tensor = torch::from_blob(chw_image.data, {3, floatImage.rows, floatImage.cols}, torch::kFloat);

  chw_image_tensor = chw_image_tensor.clone();

  if (normalize) {
    chw_image_tensor = 2.0 * chw_image_tensor - 1.0;
  }

  return chw_image_tensor;
}

inline cv::Mat tensorToCv2Mat(const torch::Tensor& tensor, bool denormalize = true) {
  torch::Tensor tensor_cpu = tensor.to(torch::kCPU).contiguous();

  const auto sizes = tensor_cpu.sizes();
  const int height = sizes[1];
  const int width = sizes[2];

  float* tensor_data = tensor_cpu.data_ptr<float>();

  cv::Mat mat(height, width, CV_32FC3, tensor_data);

  cv::Mat mat_copy = mat.clone();

  cv::cvtColor(mat_copy, mat_copy, cv::COLOR_RGB2BGR);

  if (denormalize) {
    mat_copy = (mat_copy + 1.0) / 2.0;
  }

  mat_copy.convertTo(mat_copy, CV_8UC3, 255.0);

  return mat_copy;
}

inline void saveImage(const cv::Mat& image, const std::string& filePath) {
  if (image.empty()) {
    LOG_ERROR("Error: The image is empty, cannot save to " + filePath);
    return;
  }

  try {
    bool result = cv::imwrite(filePath, image);
    if (!result) {
      LOG_ERROR("Error: Failed to save the image to " + filePath);
    }
  } catch (const cv::Exception& ex) {
    LOG_ERROR("Exception converting image to format: " + std::string(ex.what()));
  }
}

}  // namespace util
}  // namespace dmcpp
