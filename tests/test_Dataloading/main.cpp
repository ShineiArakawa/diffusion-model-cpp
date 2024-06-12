//
// Created by shinaraka on 24/06/10.
//

#include <DiffusionModelC++/Trainer/Dataloader.hpp>
#include <DiffusionModelC++/Util/FileUtil.hpp>
#include <DiffusionModelC++/Util/ImageUtil.hpp>
#include <opencv2/opencv.hpp>

int main() {
  auto dataset = dmcpp::trainer::ImageFolderDataset("/home/data/Datasets/CelebAHQ/CelebAMask-HQ/CelebA-HQ-img_train",
                                                    128,
                                                    128,
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

    for (int64_t iImage = 0; iImage < images.size(0); ++iImage) {
      const std::string& filePath = dmcpp::util::FileUtil::join("output_test_Dataloading", "sample_" + std::to_string(iImage) + ".png");

      dmcpp::util::FileUtil::mkdirs(dmcpp::util::FileUtil::dirPath(filePath));

      const cv::Mat& image = dmcpp::util::tensorToCv2Mat(images[iImage]);
      dmcpp::util::saveImage(image, filePath);
    }

    break;
  }
}
