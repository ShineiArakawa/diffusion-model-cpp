#include <omp.h>

#include <DiffusionModelC++/Trainer/Dataloader.hpp>
#include <DiffusionModelC++/Util/FileUtil.hpp>
#include <DiffusionModelC++/Util/Logging.hpp>
#include <filesystem>
#include <iostream>

namespace dmcpp::trainer {

namespace fs = std::filesystem;

ImageFolderDataset::ImageFolderDataset(const std::string& root,
                                       const int& imageWidth,
                                       const int& imageHeight,
                                       const std::string& extension)
    : _imagePaths(),
      _images(),
      _imageWidth(imageWidth),
      _imageHeight(imageHeight) {
  // Get image paths
  try {
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
      if (entry.is_regular_file() && entry.path().extension() == extension) {
        _imagePaths.push_back(entry.path().string());
      }
    }
  } catch (const fs::filesystem_error& e) {
    LOG_ERROR("Filesystem error: " + std::string(e.what()));
  } catch (const std::exception& e) {
    LOG_ERROR("General error: " + std::string(e.what()));
  }

  const int64_t nImages = static_cast<int64_t>(_imagePaths.size());

  LOG_INFO("Found " + std::to_string(nImages) + " images in " + root);

  // Load all images on memory
  LOG_INFO("Loading images ... ");

  _images.resize(nImages);

// Parallelize the image loading process using OpenMP
#pragma omp parallel for
  for (int64_t iImage = 0; iImage < nImages; ++iImage) {
    cv::Mat image = util::loadImage(_imagePaths[iImage]);
    image = util::resize(image, _imageWidth, _imageHeight);
    _images[iImage] = image;
  }

  LOG_INFO("Done.");
}

torch::data::Example<torch::Tensor> ImageFolderDataset::get(size_t index) {
  cv::Mat image = _images[index];

  if (torch::rand({1}).item<float>() < 0.5f) {
    image = util::horizontalFlip(image);
  }

  const torch::Tensor& imageTensor = util::cv2MatToTensor(image);

  return {imageTensor, imageTensor};
}

torch::optional<size_t> ImageFolderDataset::size() const {
  return _imagePaths.size();
}

}  // namespace dmcpp::trainer