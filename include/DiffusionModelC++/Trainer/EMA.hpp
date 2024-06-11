#pragma once

#include <torch/torch.h>

#include <DiffusionModelC++/Diffusion/KarrasDiffusion.hpp>
#include <unordered_map>

namespace dmcpp {
namespace trainer {

void updateEMAModel(diffusion::KarrasDiffusion& model,
                    diffusion::KarrasDiffusion& avgModel,
                    double decay);

class EMAWarmup {
 public:
  explicit EMAWarmup(double invGamma = 1.0,
                     double power = 1.0,
                     double minValue = 0.0,
                     double maxValue = 1.0,
                     int64_t startAt = 0,
                     int64_t lastEpoch = 0);

  std::unordered_map<std::string, double> state_dict() const;
  void load_state_dict(const std::unordered_map<std::string, double>& state_dict);
  double get_value() const;
  void step();

 private:
  double _invGamma;
  double _power;
  double _minValue;
  double _maxValue;
  int64_t _startAt;
  int64_t _lastEpoch;
};

}  // namespace trainer
}  // namespace dmcpp
