//
// Created by shinaraka on 24/06/10.
//

#include <DiffusionModelC++/Model/Model.hpp>
#include <DiffusionModelC++/Trainer/Dataloader.hpp>
#include <DiffusionModelC++/Util/FileUtil.hpp>
#include <DiffusionModelC++/Util/ImageUtil.hpp>
#include <opencv2/opencv.hpp>

using namespace dmcpp;

int main() {
  const int64_t nFeatures = 32;
  const int64_t nGroups = 4;
  const int64_t nHeads = 4;
  const int64_t nRepeats = 8;
  const int64_t imageSize = 32;

  const int64_t nChannels = 3LL * nRepeats;
  const int64_t nGroupsValid = nChannels / nGroups;

  model::AdaGN norm(nFeatures, nChannels, nGroupsValid);
  model::SelfAttention2D attention(nChannels, nHeads, norm);

  attention->to(torch::kCUDA);

  auto dataset = trainer::ImageFolderDataset("/home/data/Datasets/CelebAHQ/CelebAMask-HQ/CelebA-HQ-img_train",
                                             imageSize,
                                             imageSize,
                                             ".jpg");
  auto mappedDataset = dataset.map(torch::data::transforms::Stack<>());

  // DataLoader
  auto dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(mappedDataset),
      torch::data::DataLoaderOptions()
          .batch_size(32)
          .drop_last(true)
          .workers(32));

  for (auto& batch : *dataLoader) {
    torch::Tensor images = batch.data;

    images = images.to(torch::kCUDA);

    images = images.repeat({1, nRepeats, 1, 1});

    for (int64_t iImage = 0; iImage < images.size(0); ++iImage) {
      const std::string& filePath = util::FileUtil::join(util::FileUtil::join("output_test_Dataloading", "input"), "sample_" + std::to_string(iImage) + ".png");

      util::FileUtil::mkdirs(util::FileUtil::dirPath(filePath));

      const cv::Mat& image = dmcpp::util::tensorToCv2Mat(images.index({iImage, torch::indexing::Slice(0, 3), torch::indexing::Slice(), torch::indexing::Slice()}));
      util::saveImage(image, filePath);
    }

    std::cout << "images.shape : " << images.sizes() << std::endl;

    model::ConditionContext ctx;
    ctx.condition = torch::randn({images.size(0), nFeatures}, images.options());

    auto output = attention->forward(images, ctx);

    std::cout << "output.shape : " << output.sizes() << std::endl;

    for (int64_t iImage = 0; iImage < output.size(0); ++iImage) {
      const std::string& filePath = util::FileUtil::join(util::FileUtil::join("output_test_Dataloading", "output"), "sample_" + std::to_string(iImage) + ".png");

      util::FileUtil::mkdirs(util::FileUtil::dirPath(filePath));

      const cv::Mat& image = dmcpp::util::tensorToCv2Mat(output.index({iImage, torch::indexing::Slice(0, 3), torch::indexing::Slice(), torch::indexing::Slice()}));
      util::saveImage(image, filePath);
    }

    break;
  }
}
