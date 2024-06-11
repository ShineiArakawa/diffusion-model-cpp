#pragma once

#include <picojson.h>

#include <DiffusionModelC++/Util/Logging.hpp>
#include <fstream>
#include <iostream>
#include <string>

namespace dmcpp {
namespace config {

// ===============================================================================================
// Json utility
// ===============================================================================================
namespace GetValueHelpers {
template <class Type>
inline std::shared_ptr<Type> getValueFromJsonValue(const picojson::value &jsonValue) {
  return nullptr;
}

template <>
inline std::shared_ptr<int> getValueFromJsonValue<int>(const picojson::value &jsonValue) {
  std::shared_ptr<int> value = nullptr;
  if (jsonValue.is<double>()) {
    const double &doubleValue = jsonValue.get<double>();
    value = std::make_shared<int>();
    *value = (int)doubleValue;
  }
  return value;
}

template <>
inline std::shared_ptr<bool> getValueFromJsonValue<bool>(const picojson::value &jsonValue) {
  std::shared_ptr<bool> value = nullptr;
  if (jsonValue.is<bool>()) {
    value = std::make_shared<bool>();
    *value = jsonValue.get<bool>();
  }
  return value;
}

template <>
inline std::shared_ptr<std::string> getValueFromJsonValue<std::string>(const picojson::value &jsonValue) {
  std::shared_ptr<std::string> value = nullptr;
  if (jsonValue.is<std::string>()) {
    value = std::make_shared<std::string>();
    *value = jsonValue.get<std::string>();
  }
  return value;
}

template <>
inline std::shared_ptr<double> getValueFromJsonValue<double>(const picojson::value &jsonValue) {
  std::shared_ptr<double> value = nullptr;
  if (jsonValue.is<double>()) {
    const double &doubleValue = jsonValue.get<double>();
    value = std::make_shared<double>();
    *value = doubleValue;
  }
  return value;
}

template <class Type>
inline std::shared_ptr<Type> getValueFromJsonObject(const picojson::object &jsonObject) {
  picojson::value jsonValue = picojson::value(jsonObject);
  return getValueFromJsonValue<Type>(jsonValue);
}

template <class Type>
inline std::shared_ptr<Type> getScalarValue(const std::string &key, const picojson::value &jsonValue) {
  std::shared_ptr<Type> value = nullptr;
  if (jsonValue.contains(key)) {
    picojson::value childValue = jsonValue.get(key);
    value = getValueFromJsonValue<Type>(childValue);
  }
  return value;
};

template <>
inline std::shared_ptr<int> getScalarValue<int>(const std::string &key, const picojson::value &jsonValue) {
  std::shared_ptr<int> value = nullptr;
  if (jsonValue.contains(key)) {
    const picojson::value &childValue = jsonValue.get(key);
    value = getValueFromJsonValue<int>(childValue);
  }
  return value;
};

template <class EnumType>
inline EnumType parseEnum(const std::string &strEnum, const std::vector<std::string> &strElementNames) {
  EnumType enumElement = EnumType::INVALID;

  for (int i = 0; i < (int)strElementNames.size(); ++i) {
    if (strEnum == strElementNames[i]) {
      enumElement = static_cast<EnumType>(i);
      break;
    }
  }

  return enumElement;
}

template <class Type>
inline std::vector<Type *> getVectorValues(const std::string &key, const picojson::value &jsonValue) {
  std::vector<Type *> vec_values;

  if (jsonValue.contains(key)) {
    const picojson::value &childValue = jsonValue.get(key);

    if (childValue.is<picojson::array>()) {
      // 一重配列 [...]
      const picojson::array &array1 = childValue.get<picojson::array>();

      for (int i = 0; i < (int)array1.size(); i++) {
        const picojson::value &iChaildValue1 = array1[i];
        std::shared_ptr<Type> value = GetValueHelpers::getValueFromJsonValue<Type>(iChaildValue1);
        Type *ptr_value = new Type();
        *ptr_value = *value;
        vec_values.push_back(ptr_value);
      }
    } else {
      std::shared_ptr<Type> value = GetValueHelpers::getValueFromJsonValue<Type>(childValue);
      Type *ptr_value = new Type();
      *ptr_value = *value;
      vec_values.push_back(ptr_value);
    }
  } else {
    Type *ptr_value = nullptr;
    vec_values.push_back(ptr_value);
  }

  return vec_values;
}

}  // namespace GetValueHelpers

inline static const std::vector<std::string> str_DiffusionWeightingType = {"karras",
                                                                           "soft_min_snr",
                                                                           "snr"};

enum class DiffusionWeightingType {
  KARRAS,
  SOFT_MIN_SNR,
  SNR,
  INVALID
};

inline static const std::vector<std::string> str_OptimizerType = {"adamw"};

enum class OptimizerType {
  ADAMW,
  INVALID
};

inline static const std::vector<std::string> str_LRSchedulerType = {"constant"};

enum class LRSchedulerType {
  CONSTANT,
  INVALID
};

// ===============================================================================================
// Config
// ===============================================================================================
struct ModelConfig {
  int64_t inChannels = 3LL;
  int64_t inFeatures = 256LL;
  std::vector<int64_t> depth = {2, 2, 2, 4};
  std::vector<int64_t> channels = {128, 256, 256, 512};
  std::vector<bool> selfAttenDepth = {false, false, false, true};
  std::vector<bool> crossAttenDepth = {false, false, false, false};
  int64_t mappingCondDim = 0;
  int64_t unetCondDim = 0;
  int64_t crossCondDim = 0;
  double dropoutRate = 0.0;
  bool hasVariance = false;
  DiffusionWeightingType weighting = DiffusionWeightingType::KARRAS;
  double lossScale = 1.0;

  static ModelConfig load(const picojson::value &json);
};

struct DatasetConfig {
  std::string root;
  std::string extension = ".png";
  size_t batchSize = 32;
  size_t numWorkers = 4;

  static DatasetConfig load(const picojson::value &json);
};

struct OptimizerConfig {
  OptimizerType type = OptimizerType::ADAMW;
  double lr = 1e-4;
  std::vector<double> betas = {0.95, 0.999};
  double eps = 1e-6;
  double weightDecay = 1e-3;

  static OptimizerConfig load(const picojson::value &json);
};

struct LRSchedulerConfig {
  LRSchedulerType type = LRSchedulerType::CONSTANT;
  double warmup = 0.0;

  static LRSchedulerConfig load(const picojson::value &json);
};

struct EMAConfig {
  double power = 0.6667;
  double maxValue = 0.9999;

  static EMAConfig load(const picojson::value &json);
};

struct SamplerConfig {
  double sigmaData = 0.612;
  double sigmaMin = 1e-2;
  double sigmaMax = 160.0;
  double noiseDLow = 32.0;

  static SamplerConfig load(const picojson::value &json);
};

struct Config {
  ModelConfig model;
  DatasetConfig dataset;
  OptimizerConfig optimizer;
  LRSchedulerConfig lr_scheduler;
  EMAConfig ema;
  SamplerConfig sampler;

  std::string logDir{};
  int64_t seed = 1234;
  int64_t deviceID = 0;
  int64_t imageSize{};
  int64_t maxSteps = 100000;
  int64_t logEveryStep = 100;
  int64_t sampleEveryStep = 1000;

  int64_t nSamples = 16;

  static Config load(const std::string &path);
};

}  // namespace config
}  // namespace dmcpp