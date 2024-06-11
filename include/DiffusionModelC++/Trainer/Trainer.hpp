#pragma once

#include <torch/torch.h>

#include <DiffusionModelC++/Config/Config.hpp>
#include <DiffusionModelC++/Diffusion/KarrasDiffusion.hpp>
#include <DiffusionModelC++/Diffusion/Sampler.hpp>
#include <DiffusionModelC++/Trainer/Dataloader.hpp>
#include <DiffusionModelC++/Trainer/EMA.hpp>
#include <memory>

namespace dmcpp {
namespace trainer {

class Trainer {
 public:
  Trainer(const config::Config& config,
          diffusion::KarrasDiffusion& model,
          diffusion::KarrasDiffusion& emaModel);
  ~Trainer();

  void fit();

 private:
  config::Config _config;
  diffusion::KarrasDiffusion _model = nullptr;
  diffusion::KarrasDiffusion _modelEMA = nullptr;
  std::shared_ptr<torch::optim::Optimizer> _optimizer = nullptr;
  std::shared_ptr<torch::optim::LRScheduler> _lrScheduler = nullptr;
  std::shared_ptr<EMAWarmup> _EMAScheduler = nullptr;
  std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<ImageFolderDataset, torch::data::transforms::Stack<torch::data::Example<>>>, torch::data::samplers::RandomSampler>> _dataLoader = nullptr;
  std::shared_ptr<diffusion::DiffusionSampler> _sampler = nullptr;

  torch::Device _device = torch::Device(torch::kCPU);

  int64_t _step;
};

}  // namespace trainer
}  // namespace dmcpp
