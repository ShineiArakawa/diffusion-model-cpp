#pragma once

#include <torch/torch.h>

namespace dmcpp {
namespace trainer {

class ConstantLRWithWarmup : public torch::optim::LRScheduler {
 public:
  ConstantLRWithWarmup(torch::optim::Optimizer& optimizer,
                       const double warmup);

 private:
  std::vector<double> get_lrs() override;

  double _warmup;
};

}  // namespace trainer
}  // namespace dmcpp