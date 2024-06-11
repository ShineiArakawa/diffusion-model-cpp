#pragma once

#include <torch/torch.h>

namespace dmcpp {
namespace trainer {

class ImageFolderDataset : public torch::data::Dataset<ImageFolderDataset, torch::data::Example<torch::Tensor>> {
 public:
  explicit ImageFolderDataset(const std::string& root,
                              const std::string& extension = ".jpg");

  torch::data::Example<torch::Tensor> get(size_t index) override;

  torch::optional<size_t> size() const override;

 private:
  std::vector<std::string> _imagePaths;
};

}  // namespace trainer
}  // namespace dmcpp