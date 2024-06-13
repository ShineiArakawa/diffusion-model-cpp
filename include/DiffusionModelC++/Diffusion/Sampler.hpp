//
// Created by shinaraka on 24/06/11.
//

#pragma once

#include <torch/torch.h>

#include <DiffusionModelC++/Config/Config.hpp>
#include <DiffusionModelC++/Diffusion/KarrasDiffusion.hpp>

namespace dmcpp {
namespace diffusion {
inline torch::Tensor appendZero(const torch::Tensor& x) {
  return torch::cat({x, torch::zeros({1}, x.options())});
};

// =========================================================================================================
// Samplers for sampling
// =========================================================================================================
inline torch::Tensor getSigmasKarras(int n,
                                     double sigma_min,
                                     double sigma_max,
                                     double rho = 7.0,
                                     torch::Device device = torch::kCPU) {
  torch::Tensor ramp = torch::linspace(0.0, 1.0, n, torch::TensorOptions().device(device));

  const double min_inv_rho = std::pow(sigma_min, 1.0 / rho);
  const double max_inv_rho = std::pow(sigma_max, 1.0 / rho);

  torch::Tensor sigmas = torch::pow(max_inv_rho + ramp * (min_inv_rho - max_inv_rho), rho);

  // 最後にゼロを追加
  sigmas = appendZero(sigmas);

  return sigmas;
};

inline torch::Tensor sample_heun(
    KarrasDiffusion& model,
    torch::Tensor x,
    torch::Tensor sigmas,
    float s_churn = 0.0,
    float s_tmin = 0.0,
    float s_tmax = std::numeric_limits<float>::infinity(),
    float s_noise = 1.0) {
  torch::NoGradGuard no_grad;

  const torch::Tensor s_in = torch::ones({x.size(0)}, x.options());

  model::ImageUNetModelForwardArgs args;

  for (int i = 0; i < sigmas.size(0) - 1; ++i) {
    const float iSigma = sigmas[i].item<float>();

    const float gamma = (s_tmin <= iSigma && iSigma <= s_tmax) ? std::min(s_churn / (sigmas.size(0) - 1.0f), std::sqrt(2.0f) - 1.0f) : 0.0f;

    const torch::Tensor eps = torch::randn_like(x) * s_noise;

    const torch::Tensor sigma_hat = sigmas[i] * (gamma + 1.0f);

    if (gamma > 0.0f) {
      x = x + eps * torch::sqrt(sigma_hat * sigma_hat - iSigma * iSigma);
    }

    const torch::Tensor& denoised = model->forward(x, sigma_hat * s_in, args);
    const torch::Tensor& d = KarrasDiffusionImpl::toD(x, sigma_hat * s_in, denoised);

    const torch::Tensor dt = sigmas[i + 1] - sigma_hat;

    if (sigmas[i + 1].item<float>() == 0) {
      x = x + d * dt;
    } else {
      const auto x_2 = x + d * dt;
      const torch::Tensor& denoised_2 = model->forward(x_2, sigmas[i + 1] * s_in, args);
      const auto d_2 = KarrasDiffusionImpl::toD(x_2, sigmas[i + 1] * s_in, denoised_2);
      const auto d_prime = (d + d_2) / 2.0;
      x = x + d_prime * dt;
    }
  }

  return x;
};

// =========================================================================================================
// Samplers for training
// =========================================================================================================

inline torch::Tensor logSNRScheduleCosine(const torch::Tensor& t,
                                          double logsnr_min,
                                          double logsnr_max) {
  double t_min = std::atan(std::exp(-0.5 * logsnr_max));
  double t_max = std::atan(std::exp(-0.5 * logsnr_min));
  return -2.0 * torch::log(torch::tan(t_min + t * (t_max - t_min)));
}

inline torch::Tensor logSNRScheduleCosineShifted(const torch::Tensor& t,
                                                 double image_d,
                                                 double noise_d,
                                                 double logsnr_min,
                                                 double logsnr_max) {
  double shift = 2.0 * std::log(noise_d / image_d);
  return logSNRScheduleCosine(t, logsnr_min - shift, logsnr_max - shift) + shift;
}

inline torch::Tensor logSNRScheduleCosineInterpolated(const torch::Tensor& t,
                                                      double image_d,
                                                      double noise_d_low,
                                                      double noise_d_high,
                                                      double logsnr_min,
                                                      double logsnr_max) {
  const torch::Tensor& logsnr_low = logSNRScheduleCosineShifted(t, image_d, noise_d_low, logsnr_min, logsnr_max);
  const torch::Tensor& logsnr_high = logSNRScheduleCosineShifted(t, image_d, noise_d_high, logsnr_min, logsnr_max);
  return torch::lerp(logsnr_low, logsnr_high, t);
}

inline torch::Tensor randCosineInterpolated(const at::IntArrayRef& shape,
                                            double image_d,
                                            double noise_d_low,
                                            double noise_d_high,
                                            double sigma_data = 1.0,
                                            double min_value = 1e-3,
                                            double max_value = 1e3,
                                            torch::Device device = torch::kCPU,
                                            torch::Dtype dtype = torch::kFloat32) {
  double logsnr_min = -2.0 * std::log(min_value / sigma_data);
  double logsnr_max = -2.0 * std::log(max_value / sigma_data);
  const torch::Tensor& u = torch::rand(shape, torch::TensorOptions().device(device).dtype(dtype));
  const torch::Tensor& logsnr = logSNRScheduleCosineInterpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max);
  return torch::exp(-logsnr / 2.0) * sigma_data;
}

class DiffusionSampler {
 public:
  virtual ~DiffusionSampler() = default;

  virtual torch::Tensor sample(const at::IntArrayRef& shape,
                               const torch::Device& device,
                               const torch::Dtype& dtype) = 0;
};

class CosineInterpolatedSampler : public DiffusionSampler {
 public:
  explicit CosineInterpolatedSampler(const config::Config& config);
  ~CosineInterpolatedSampler() override;

  torch::Tensor sample(const at::IntArrayRef& shape,
                       const torch::Device& device,
                       const torch::Dtype& dtype) override;

 private:
  double _imageD;
  double _noiseDLow;
  double _noiseDHigh;
  double _sigmaData;
  double _minValue;
  double _maxValue;
};

}  // namespace diffusion
}  // namespace dmcpp