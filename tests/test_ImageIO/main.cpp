//
// Created by shinaraka on 24/06/10.
//

#include <DiffusionModelC++/Util/ImageUtil.hpp>

int main() {
  auto image = dmcpp::util::loadImage("/home/data/Datasets/CelebAHQ/CelebAMask-HQ/CelebA-HQ-img_train/all/0.jpg");

  auto imageTensor = dmcpp::util::cv2MatToTensor(image);

  auto imageOut = dmcpp::util::tensorToCv2Mat(imageTensor);

  // imageOut = dmcpp::util::horizontalFlip(imageOut);

  // imageOut = dmcpp::util::verticalFlip(imageOut);

  imageOut = dmcpp::util::resize(imageOut, 128, 128);

  dmcpp::util::saveImage(imageOut, "output.png");
}
