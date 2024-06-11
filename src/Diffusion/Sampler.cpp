//
// Created by shinaraka on 24/06/11.
//

#include <DiffusionModelC++/Diffusion/Sampler.hpp>

namespace dmcpp::diffusion {

CosineInterpolatedSampler::CosineInterpolatedSampler(const config::Config &config) {
  _imageD = static_cast<double>(config.imageSize);
  _noiseDLow = config.sampler.noiseDLow;
  _noiseDHigh = static_cast<double>(config.imageSize);
  _sigmaData = config.sampler.sigmaData;
  _minValue = std::min(config.sampler.sigmaMin, 1e-3);
  _maxValue = std::max(config.sampler.sigmaMax, 1e3);
}

CosineInterpolatedSampler::~CosineInterpolatedSampler() = default;

torch::Tensor CosineInterpolatedSampler::sample(const at::IntArrayRef &shape,
                                                const torch::Device &device,
                                                const torch::Dtype &dtype) {
  return randCosineInterpolated(
      shape,
      _imageD,
      _noiseDLow,
      _noiseDHigh,
      _sigmaData,
      _minValue,
      _maxValue,
      device,
      dtype);
}

}  // namespace dmcpp::diffusion