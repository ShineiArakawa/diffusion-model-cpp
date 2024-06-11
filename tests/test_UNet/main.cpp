//
// Created by shinaraka on 24/06/10.
//

#include <DiffusionModelC++/Model/Model.hpp>

int main() {
  const std::vector<int64_t> depth = {2, 2, 4};
  const std::vector<int64_t> channels = {128, 256, 256};
  const std::vector<bool> selfAttenDepth = {false, true, true};
  const std::vector<bool> crossAttenDepth = {false, false, false};

  dmcpp::model::ImageUNetModel model(3,
                                     256,
                                     depth,
                                     channels,
                                     selfAttenDepth,
                                     crossAttenDepth);

  std::cout << model << std::endl;

  model->to(torch::kCUDA);

  dmcpp::model::ImageUNetModelForwardArgs args;

  const at::Tensor& input = torch::randn({16, 3, 64, 64}).to(torch::kCUDA);
  const at::Tensor& sigma = torch::randn({16}).to(torch::kCUDA);

  dmcpp::model::ImageUNetModelForwardReturn ret = model->forward(input, sigma, args);
}
