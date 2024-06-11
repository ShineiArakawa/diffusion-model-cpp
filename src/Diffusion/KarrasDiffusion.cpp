#include <DiffusionModelC++/Diffusion/KarrasDiffusion.hpp>

namespace dmcpp::diffusion {

KarrasDiffusionImpl::KarrasDiffusionImpl(model::ImageUNetModel& innerModel,
                                         float sigmaData,
                                         config::DiffusionWeightingType weighting,
                                         float scales)
    : _innerModel(innerModel),
      _sigmaData(sigmaData),
      _scales(scales) {
  switch (weighting) {
    case config::DiffusionWeightingType::KARRAS:
      _weightingFunc = [](const torch::Tensor& sigma) {
        return torch::ones_like(sigma);
      };
      break;
    case config::DiffusionWeightingType::SOFT_MIN_SNR:
      _weightingFunc = [this](const torch::Tensor& sigma) {
        return (sigma * _sigmaData).pow(2) / (sigma.pow(2) + _sigmaData * _sigmaData).pow(2);
      };
      break;
    case config::DiffusionWeightingType::SNR:
      _weightingFunc = [this](const torch::Tensor& sigma) {
        const float sigmaDataSquared = _sigmaData * _sigmaData;
        return sigmaDataSquared / (sigma.pow(2) + sigmaDataSquared);
      };
      break;
    case config::DiffusionWeightingType::INVALID:
      LOG_CRITICAL("Invalid DiffusionWeightingType");
      exit(EXIT_FAILURE);
  }

  register_module("innerModel", _innerModel);
}

void KarrasDiffusionImpl::getScaling(const torch::Tensor& sigma, torch::Tensor& skip, torch::Tensor& out, torch::Tensor& in) const {
  const float& sigmaDataSquared = _sigmaData * _sigmaData;
  const torch::Tensor& sigmaSquared = sigma.pow(2);

  const torch::Tensor& sigmaDenominator = sigmaSquared + sigmaDataSquared;
  const torch::Tensor& sigmaDenominatorSqrt = sigmaDenominator.sqrt();

  skip = sigmaDataSquared / sigmaDenominator;
  out = sigma * _sigmaData / sigmaDenominatorSqrt;
  in = 1.0 / sigmaDenominatorSqrt;
}

torch::Tensor KarrasDiffusionImpl::loss(const torch::Tensor& input,
                                        const torch::Tensor& noise,
                                        const torch::Tensor& sigma,
                                        const model::ImageUNetModelForwardArgs& args) {
  const int64_t b = input.size(0);

  const torch::Tensor& noisedInput = input + noise * sigma.view({b, 1, 1, 1});

  const torch::Tensor& denoised = forward(noisedInput, sigma, args);

  const torch::Tensor& eps = toD(noisedInput, sigma, denoised);

  return (eps - noise).pow(2).flatten(1).mean(1);
}

torch::Tensor KarrasDiffusionImpl::forward(const torch::Tensor& input,
                                           const torch::Tensor& sigma,
                                           const model::ImageUNetModelForwardArgs& args) {
  const int64_t b = input.size(0);

  torch::Tensor skip, out, in;
  getScaling(sigma, skip, out, in);

  skip = skip.view({b, 1, 1, 1});
  out = out.view({b, 1, 1, 1});
  in = in.view({b, 1, 1, 1});

  const model::ImageUNetModelForwardReturn& modelReturn = _innerModel->forward(input * in, sigma, args);

  return modelReturn.output * out + input * skip;
}

void KarrasDiffusionImpl::reset() {
  _innerModel->reset();
}

torch::Tensor KarrasDiffusionImpl::toD(const torch::Tensor& x,
                                       const torch::Tensor& sigma,
                                       const torch::Tensor& denoised) {
  const int64_t b = x.size(0);
  return (x - denoised) / sigma.view({b, 1, 1, 1});
}

}  // namespace dmcpp::diffusion