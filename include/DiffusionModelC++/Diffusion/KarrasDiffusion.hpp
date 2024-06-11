#pragma once

#include <torch/torch.h>

#include <DiffusionModelC++/Config/Config.hpp>
#include <DiffusionModelC++/Model/Model.hpp>
#include <functional>

namespace dmcpp {
namespace diffusion {

struct KarrasDiffusionImpl : public torch::nn::Cloneable<KarrasDiffusionImpl> {
  explicit KarrasDiffusionImpl(model::ImageUNetModel& innerModel,
                               float sigmaData = 1.0f,
                               config::DiffusionWeightingType weighting = config::DiffusionWeightingType::KARRAS,
                               float scales = 1.0f);

  void getScaling(const torch::Tensor& sigma, torch::Tensor& skip, torch::Tensor& out, torch::Tensor& in) const;

  torch::Tensor loss(const torch::Tensor& input,
                     const torch::Tensor& noise,
                     const torch::Tensor& sigma,
                     const model::ImageUNetModelForwardArgs& args);

  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& sigma,
                        const model::ImageUNetModelForwardArgs& args);

  void reset() override;

  static torch::Tensor toD(const torch::Tensor& x,
                           const torch::Tensor& sigma,
                           const torch::Tensor& denoised);

 private:
  model::ImageUNetModel _innerModel = nullptr;
  float _sigmaData;
  float _scales;
  std::function<torch::Tensor(const torch::Tensor&)> _weightingFunc = nullptr;
};

TORCH_MODULE(KarrasDiffusion);

}  // namespace diffusion

inline diffusion::KarrasDiffusion getDiffusionModel(const config::Config& config) {
  model::ImageUNetModel innerModel(config.model.inChannels,
                                   config.model.inFeatures,
                                   config.model.depth,
                                   config.model.channels,
                                   config.model.selfAttenDepth,
                                   config.model.crossAttenDepth,
                                   config.model.mappingCondDim,
                                   config.model.unetCondDim,
                                   config.model.crossCondDim,
                                   config.model.dropoutRate,
                                   config.model.hasVariance);

  // Diffusion model
  return diffusion::KarrasDiffusion(innerModel,
                                    config.sampler.sigmaData,
                                    config.model.weighting,
                                    config.model.lossScale);
}

}  // namespace dmcpp
