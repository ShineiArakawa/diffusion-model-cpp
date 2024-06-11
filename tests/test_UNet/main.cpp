//
// Created by shinaraka on 24/06/10.
//

#include <DiffusionModelC++/Model/Model.hpp>
#include <DiffusionModelC++/Trainer/Dataloader.hpp>
#include <DiffusionModelC++/Util/FileUtil.hpp>
#include <DiffusionModelC++/Util/ImageUtil.hpp>

using namespace dmcpp;

int main() {
  const std::vector<int64_t> depth = {2, 2, 4};
  const std::vector<int64_t> channels = {128, 256, 256};
  const std::vector<bool> selfAttenDepth = {false, false, false};
  const std::vector<bool> crossAttenDepth = {false, false, false};
  const int64_t imageSize = 128;
  const int64_t batchSize = 4;

  model::ImageUNetModel unet(3,
                             256,
                             depth,
                             channels,
                             selfAttenDepth,
                             crossAttenDepth);

  std::cout << unet << std::endl;

  torch::nn::init::normal_(unet->_outProj->weight);

  unet->to(torch::kCUDA);

  model::ImageUNetModelForwardArgs args;

  auto dataset = trainer::ImageFolderDataset("/home/data/Datasets/CelebAHQ/CelebAMask-HQ/CelebA-HQ-img_train",
                                             imageSize,
                                             imageSize,
                                             ".jpg");
  auto mappedDataset = dataset.map(torch::data::transforms::Stack<>());

  // DataLoader
  auto dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(mappedDataset),
      torch::data::DataLoaderOptions()
          .batch_size(batchSize)
          .drop_last(true)
          .workers(32));

  for (auto& batch : *dataLoader) {
    torch::Tensor images = batch.data;

    images = images.to(torch::kCUDA);

    for (int64_t iImage = 0; iImage < images.size(0); ++iImage) {
      const std::string& filePath = util::FileUtil::join(util::FileUtil::join("output_test_UNet", "input"), "sample_" + std::to_string(iImage) + ".png");

      util::FileUtil::mkdirs(util::FileUtil::dirPath(filePath));

      const cv::Mat& image = dmcpp::util::tensorToCv2Mat(images.index({iImage, torch::indexing::Slice(0, 3), torch::indexing::Slice(), torch::indexing::Slice()}));
      util::saveImage(image, filePath);
    }

    const at::Tensor& sigma = torch::randn({images.size(0)}).to(torch::kCUDA);
    dmcpp::model::ImageUNetModelForwardReturn ret = unet->forward(images, sigma, args);

    auto output = ret.output;

    for (int64_t iImage = 0; iImage < output.size(0); ++iImage) {
      const std::string& filePath = util::FileUtil::join(util::FileUtil::join("output_test_UNet", "output"), "sample_" + std::to_string(iImage) + ".png");

      util::FileUtil::mkdirs(util::FileUtil::dirPath(filePath));

      const cv::Mat& image = dmcpp::util::tensorToCv2Mat(output.index({iImage, torch::indexing::Slice(0, 3), torch::indexing::Slice(), torch::indexing::Slice()}));
      util::saveImage(image, filePath);
    }

    break;
  }
}
