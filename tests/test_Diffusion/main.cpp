//
// Created by shinaraka on 24/06/10.
//

#include <torch/torch.h>

#include <DiffusionModelC++/Diffusion/Sampler.hpp>
#include <functional>
#include <iostream>

using namespace dmcpp;

using Model_t = std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>;

inline torch::Tensor sample_heun(
    Model_t& model,
    torch::Tensor x,
    torch::Tensor sigmas,
    float s_churn = 0.0,
    float s_tmin = 0.0,
    float s_tmax = std::numeric_limits<float>::infinity(),
    float s_noise = 1.0) {
  torch::NoGradGuard no_grad;

  const torch::Tensor s_in = torch::ones({x.size(0)}, x.options());

  for (int i = 0; i < sigmas.size(0) - 1; ++i) {
    const float iSigma = sigmas[i].item<float>();

    const float gamma = (s_tmin <= iSigma && iSigma <= s_tmax) ? std::min(s_churn / (sigmas.size(0) - 1.0f), std::sqrt(2.0f) - 1.0f) : 0.0f;

    const torch::Tensor eps = torch::randn_like(x) * s_noise;

    const torch::Tensor sigma_hat = sigmas[i] * (gamma + 1.0f);

    if (gamma > 0.0f) {
      x = x + eps * torch::sqrt(sigma_hat * sigma_hat - iSigma * iSigma);
    }

    const torch::Tensor& denoised = model(x, sigma_hat * s_in);
    const torch::Tensor& d = diffusion::KarrasDiffusionImpl::toD(x, sigma_hat * s_in, denoised);

    const torch::Tensor dt = sigmas[i + 1] - sigma_hat;

    if (sigmas[i + 1].item<float>() == 0) {
      x = x + d * dt;
    } else {
      const auto x_2 = x + d * dt;
      const torch::Tensor& denoised_2 = model(x_2, sigmas[i + 1] * s_in);
      const auto d_2 = diffusion::KarrasDiffusionImpl::toD(x_2, sigmas[i + 1] * s_in, denoised_2);
      const auto d_prime = (d + d_2) / 2.0;
      x = x + d_prime * dt;
    }
  }

  return x;
};

void test_getSigmasKarras() {
  torch::Tensor x = diffusion::getSigmasKarras(50, 1e-2, 80.0, 7.0);
  std::cout << x << std::endl;
};

void test_sample_heun() {
  torch::Tensor sigmas = diffusion::getSigmasKarras(50, 1e-2, 80.0, 7.0);
  torch::Tensor x = torch::randn({4, 3, 16, 16});

  Model_t model = [](const torch::Tensor& x_, const torch::Tensor& sigma_) {
    return x_ + torch::randn_like(x_) * sigma_.view({x_.size(0), 1, 1, 1});
  };

  torch::Tensor y = sample_heun(model, x, sigmas);
  std::cout << y[0] << std::endl;
}

int main(int argc, char* argv[]) {
  torch::manual_seed(0);

  // test_getSigmasKarras();
  test_sample_heun();
}
