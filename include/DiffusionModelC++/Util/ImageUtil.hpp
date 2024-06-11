#pragma once

#include <torch/torch.h>

#include <DiffusionModelC++/Util/FileUtil.hpp>
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
  image.convertTo(floatImage, CV_32F, 1.0 / 255.0);

  // NOTE: BGR -> RGB
  cv::Mat floatRGBImage;
  cv::cvtColor(floatImage, floatRGBImage, cv::COLOR_BGR2RGB);

  TORCH_CHECK(floatRGBImage.channels() == 3, "Input channel != 3");

  at::Tensor tensorImage = torch::from_blob(floatRGBImage.data, {floatRGBImage.rows, floatRGBImage.cols, 3}, torch::kFloat32);

  // NOTE: HWC (Height, Width, Channels) -> CHW (Channels, Height, Width)
  tensorImage = tensorImage.permute({2, 0, 1});

  if (normalize) {
    tensorImage = 2.0 * tensorImage - 1.0;
  }

  return tensorImage.clone();
}

inline cv::Mat tensorToCv2Mat(const torch::Tensor& tensor, bool denormalize = true) {
  torch::Tensor tensor_cpu = tensor.to(torch::kCPU).permute({1, 2, 0}).contiguous();

  if (denormalize) {
    tensor_cpu = (tensor_cpu + 1.0) / 2.0;
  }

  const auto sizes = tensor_cpu.sizes();
  const int height = sizes[0];
  const int width = sizes[1];
  const int channels = sizes[2];

  TORCH_CHECK(channels == 3, "Input channel != 3");

  cv::Mat mat(height, width, CV_32FC3, tensor_cpu.data_ptr<float>());
  mat.convertTo(mat, CV_8UC3, 255.0);

  cv::Mat matBgr;
  cv::cvtColor(mat, matBgr, cv::COLOR_RGB2BGR);

  return matBgr.clone();
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

inline cv::Mat resize(const cv::Mat& image, const int width, const int height) {
  cv::Size new_size(width, height);

  cv::Mat dst;

  cv::resize(image, dst, new_size, 0.0, 0.0, cv::INTER_LANCZOS4);

  return dst;
}

inline cv::Mat horizontalFlip(const cv::Mat& image) {
  cv::Mat dst;

  cv::flip(image, dst, 1);

  return dst;
}

inline cv::Mat verticalFlip(const cv::Mat& image) {
  cv::Mat dst;

  cv::flip(image, dst, 0);

  return dst;
}

inline void DEBUG_saveImages(const torch::Tensor& images, const std::string& dirPath) {
  util::FileUtil::mkdirs(dirPath);

  for (int64_t iImage = 0; iImage < images.size(0); ++iImage) {
    const std::string& filePath = util::FileUtil::join(dirPath, "sample_" + std::to_string(iImage) + ".png");

    const cv::Mat& image = dmcpp::util::tensorToCv2Mat(images.index({iImage, torch::indexing::Slice(0, 3), torch::indexing::Slice(), torch::indexing::Slice()}));
    util::saveImage(image, filePath);
  }
}

}  // namespace util
}  // namespace dmcpp
