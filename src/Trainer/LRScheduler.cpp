#include <DiffusionModelC++/Trainer/LRScheduler.hpp>

namespace dmcpp::trainer {

ConstantLRWithWarmup::ConstantLRWithWarmup(torch::optim::Optimizer& optimizer,
                                           const double warmup)
    : torch::optim::LRScheduler(optimizer),
      _warmup(warmup) {
}

std::vector<double> ConstantLRWithWarmup::get_lrs() {
  const double warmup = 1.0 - std::pow(_warmup, step_count_ + 1);

  const auto& current_lrs = get_current_lrs();
  std::vector<double> lrs(current_lrs.size());

  for (size_t i = 0; i < lrs.size(); ++i) {
    lrs[i] = current_lrs[i] * warmup;
  }

  return lrs;
}

}  // namespace dmcpp::trainer
