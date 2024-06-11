#include <DiffusionModelC++/Trainer/Dataloader.hpp>
#include <DiffusionModelC++/Util/FileUtil.hpp>
#include <DiffusionModelC++/Util/ImageUtil.hpp>
#include <DiffusionModelC++/Util/Logging.hpp>
#include <filesystem>
#include <iostream>

namespace dmcpp::trainer {

namespace fs = std::filesystem;

ImageFolderDataset::ImageFolderDataset(const std::string& root,
                                       const std::string& extension)
    : _imagePaths() {
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

  LOG_INFO("Found " + std::to_string(_imagePaths.size()) + " images in " + root);
}

torch::data::Example<torch::Tensor> ImageFolderDataset::get(size_t index) {
  const cv::Mat& image = util::loadImage(_imagePaths[index]);
  const torch::Tensor& imageTensor = util::cv2MatToTensor(image);

  return {imageTensor, imageTensor};
}

torch::optional<size_t> ImageFolderDataset::size() const {
  return _imagePaths.size();
}

}  // namespace dmcpp::trainer