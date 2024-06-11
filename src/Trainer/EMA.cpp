#include <DiffusionModelC++/Trainer/EMA.hpp>

namespace dmcpp::trainer {

void updateEMAModel(diffusion::KarrasDiffusion& model,
                    diffusion::KarrasDiffusion& avgModel,
                    double decay) {
  // Update param
  auto model_params = model->named_parameters();
  auto averaged_params = avgModel->named_parameters();
  TORCH_CHECK(model_params.size() == averaged_params.size(), "Model parameters size mismatch!");

  for (const auto& param_pair : model_params) {
    auto name = param_pair.key();
    auto param = param_pair.value();
    auto avg_param = averaged_params[name];
    avg_param.lerp_(param, 1.0 - decay);
  }

  // Update buffer
  auto model_buffers = model->named_buffers();
  auto averaged_buffers = avgModel->named_buffers();
  TORCH_CHECK(model_buffers.size() == averaged_buffers.size(), "Model buffers size mismatch!");

  for (const auto& buffer_pair : model_buffers) {
    auto name = buffer_pair.key();
    auto buf = buffer_pair.value();
    auto avg_buf = averaged_buffers[name];
    avg_buf.copy_(buf);
  }
}

EMAWarmup::EMAWarmup(double invGamma,
                     double power,
                     double minValue,
                     double maxValue,
                     int64_t startAt,
                     int64_t lastEpoch)
    : _invGamma(invGamma),
      _power(power),
      _minValue(minValue),
      _maxValue(maxValue),
      _startAt(startAt),
      _lastEpoch(lastEpoch) {}

std::unordered_map<std::string, double> EMAWarmup::state_dict() const {
  return {
      {"invGamma", _invGamma},
      {"power", _power},
      {"minValue", _minValue},
      {"maxValue", _maxValue},
      {"startAt", static_cast<double>(_startAt)},
      {"lastEpoch", static_cast<double>(_lastEpoch)}};
}

void EMAWarmup::load_state_dict(const std::unordered_map<std::string, double>& state_dict) {
  _invGamma = state_dict.at("invGamma");
  _power = state_dict.at("power");
  _minValue = state_dict.at("minValue");
  _maxValue = state_dict.at("maxValue");
  _startAt = static_cast<int64_t>(state_dict.at("startAt"));
  _lastEpoch = static_cast<int64_t>(state_dict.at("lastEpoch"));
}

double EMAWarmup::get_value() const {
  int64_t epoch = _lastEpoch - _startAt;

  if (epoch < 0LL) {
    epoch = 0LL;
  }

  const double value = 1.0 - std::pow(1.0 + epoch / _invGamma, -_power);

  return (epoch < 0LL) ? 0.0 : std::min(_maxValue, std::max(_minValue, value));
}

void EMAWarmup::step() {
  ++_lastEpoch;
}

}  // namespace dmcpp::trainer