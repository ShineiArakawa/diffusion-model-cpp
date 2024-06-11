//
// Created by shinaraka on 24/06/10.
//

#include <DiffusionModelC++/Util/ImageUtil.hpp>

int main() {
  auto image = dmcpp::util::loadImage("/home/data/DockerProjects/NvidiaPytorch/Datasets/CelebA/celeba128_train/all/000002.png");

  auto imageTensor = dmcpp::util::cv2MatToTensor(image);

  auto imageOut = dmcpp::util::tensorToCv2Mat(imageTensor);

  dmcpp::util::saveImage(imageOut, "output.png");
}
