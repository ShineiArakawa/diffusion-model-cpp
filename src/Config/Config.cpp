#include <DiffusionModelC++/Config/Config.hpp>

namespace dmcpp::config {
ModelConfig ModelConfig::load(const picojson::value &json) {
  ModelConfig config;

  {
    const auto ptr = GetValueHelpers::getScalarValue<int>("in_channels", json);
    if (ptr != nullptr) {
      config.inChannels = static_cast<int64_t>(*ptr);
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<int>("in_features", json);
    if (ptr != nullptr) {
      config.inFeatures = static_cast<int64_t>(*ptr);
    }
  }

  {
    const auto ptr = GetValueHelpers::getVectorValues<int>("depth", json);

    if (ptr[0] != nullptr) {
      std::vector<int64_t> depth;

      for (auto ptr_element : ptr) {
        if (ptr_element != nullptr) {
          depth.push_back(static_cast<int64_t>(*ptr_element));
        }
      }

      config.depth = depth;
    }
  }

  {
    const auto ptr = GetValueHelpers::getVectorValues<int>("channels", json);

    if (ptr[0] != nullptr) {
      std::vector<int64_t> channels;

      for (auto ptr_element : ptr) {
        if (ptr_element != nullptr) {
          channels.push_back(static_cast<int64_t>(*ptr_element));
        }
      }

      config.channels = channels;
    }
  }

  {
    const auto ptr = GetValueHelpers::getVectorValues<bool>("self_atten_depth", json);

    if (ptr[0] != nullptr) {
      std::vector<bool> selfAttenDepth;

      for (auto ptr_element : ptr) {
        if (ptr_element != nullptr) {
          selfAttenDepth.push_back(*ptr_element);
        }
      }

      config.selfAttenDepth = selfAttenDepth;
    }
  }

  {
    const auto ptr = GetValueHelpers::getVectorValues<bool>("cross_atten_depth", json);

    if (ptr[0] != nullptr) {
      std::vector<bool> crossAttenDepth;

      for (auto ptr_element : ptr) {
        if (ptr_element != nullptr) {
          crossAttenDepth.push_back(*ptr_element);
        }
      }

      config.crossAttenDepth = crossAttenDepth;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<int>("mapping_cond_dim", json);
    if (ptr != nullptr) {
      config.mappingCondDim = static_cast<int64_t>(*ptr);
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<int>("unet_cond_dim", json);
    if (ptr != nullptr) {
      config.unetCondDim = static_cast<int64_t>(*ptr);
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<int>("cross_cond_dim", json);
    if (ptr != nullptr) {
      config.crossCondDim = static_cast<int64_t>(*ptr);
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("dropout_rate", json);
    if (ptr != nullptr) {
      config.dropoutRate = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<bool>("has_variance", json);
    if (ptr != nullptr) {
      config.hasVariance = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<std::string>("weighting", json);
    if (ptr != nullptr) {
      config.weighting = GetValueHelpers::parseEnum<DiffusionWeightingType>(*ptr, str_DiffusionWeightingType);
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("loss_scale", json);
    if (ptr != nullptr) {
      config.lossScale = *ptr;
    }
  }

  return config;
}

DatasetConfig DatasetConfig::load(const picojson::value &json) {
  DatasetConfig config;

  {
    const auto ptr = GetValueHelpers::getScalarValue<std::string>("root", json);
    if (ptr != nullptr) {
      config.root = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<std::string>("extension", json);
    if (ptr != nullptr) {
      config.extension = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<int>("batch_size", json);
    if (ptr != nullptr) {
      config.batchSize = static_cast<size_t>(*ptr);
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<int>("num_workers", json);
    if (ptr != nullptr) {
      config.numWorkers = static_cast<size_t>(*ptr);
    }
  }

  return config;
}

OptimizerConfig OptimizerConfig::load(const picojson::value &json) {
  OptimizerConfig config;

  {
    const auto ptr = GetValueHelpers::getScalarValue<std::string>("type", json);
    if (ptr != nullptr) {
      config.type = GetValueHelpers::parseEnum<OptimizerType>(*ptr, str_OptimizerType);
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("lr", json);
    if (ptr != nullptr) {
      config.lr = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getVectorValues<double>("betas", json);

    if (ptr[0] != nullptr) {
      std::vector<double> betas;

      for (auto ptr_element : ptr) {
        if (ptr_element != nullptr) {
          betas.push_back(*ptr_element);
        }
      }

      config.betas = betas;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("eps", json);
    if (ptr != nullptr) {
      config.eps = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("weight_decay", json);
    if (ptr != nullptr) {
      config.weightDecay = *ptr;
    }
  }

  return config;
}

LRSchedulerConfig LRSchedulerConfig::load(const picojson::value &json) {
  LRSchedulerConfig config;

  {
    const auto ptr = GetValueHelpers::getScalarValue<std::string>("type", json);
    if (ptr != nullptr) {
      config.type = GetValueHelpers::parseEnum<LRSchedulerType>(*ptr, str_LRSchedulerType);
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("warmup", json);
    if (ptr != nullptr) {
      config.warmup = *ptr;
    }
  }

  return config;
}

EMAConfig EMAConfig::load(const picojson::value &json) {
  EMAConfig config;

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("power", json);
    if (ptr != nullptr) {
      config.power = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("max_value", json);
    if (ptr != nullptr) {
      config.maxValue = *ptr;
    }
  }

  return config;
}

SamplerConfig SamplerConfig::load(const picojson::value &json) {
  SamplerConfig config;

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("sigma_data", json);
    if (ptr != nullptr) {
      config.sigmaData = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("sigma_min", json);
    if (ptr != nullptr) {
      config.sigmaMin = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("sigma_max", json);
    if (ptr != nullptr) {
      config.sigmaMax = *ptr;
    }
  }

  {
    const auto ptr = GetValueHelpers::getScalarValue<double>("noise_d_low", json);
    if (ptr != nullptr) {
      config.noiseDLow = *ptr;
    }
  }

  return config;
}

Config Config::load(const std::string &path) {
  LOG_INFO("Load config file: " + path);

  std::shared_ptr<picojson::value> jsonValue = std::make_shared<picojson::value>();

  if (auto fs = std::ifstream(path, std::ios::binary)) {
    fs >> (*jsonValue);
    fs.close();
  } else {
    LOG_CRITICAL("Failed to open parameters file: " + path);
    exit(EXIT_FAILURE);
  }

  Config config;

  // Subconfig
  if (jsonValue->contains("model")) {
    config.model = ModelConfig::load(jsonValue->get("model"));
  }

  if (jsonValue->contains("dataset")) {
    config.dataset = DatasetConfig::load(jsonValue->get("dataset"));
  }

  if (jsonValue->contains("optim")) {
    config.optimizer = OptimizerConfig::load(jsonValue->get("optim"));
  }

  if (jsonValue->contains("lr_sched")) {
    config.lr_scheduler = LRSchedulerConfig::load(jsonValue->get("lr_sched"));
  }

  if (jsonValue->contains("ema")) {
    config.ema = EMAConfig::load(jsonValue->get("ema"));
  }

  if (jsonValue->contains("sampler")) {
    config.sampler = SamplerConfig::load(jsonValue->get("sampler"));
  }

  // value
  {
    const auto ptr_logDir = GetValueHelpers::getScalarValue<std::string>("log_dir", *jsonValue);
    if (ptr_logDir != nullptr) {
      config.logDir = *ptr_logDir;
    } else {
      LOG_ERROR("'log_dir' is not specified.");
      exit(EXIT_FAILURE);
    }
  }

  {
    const auto ptr_seed = GetValueHelpers::getScalarValue<int>("seed", *jsonValue);
    if (ptr_seed != nullptr) {
      config.seed = static_cast<int64_t>(*ptr_seed);
    }
  }

  {
    const auto ptr_deviceID = GetValueHelpers::getScalarValue<int>("device_ID", *jsonValue);
    if (ptr_deviceID != nullptr) {
      config.deviceID = static_cast<int64_t>(*ptr_deviceID);
    }
  }

  {
    const auto ptr_imageSize = GetValueHelpers::getScalarValue<int>("image_size", *jsonValue);
    if (ptr_imageSize != nullptr) {
      config.imageSize = static_cast<int64_t>(*ptr_imageSize);
    } else {
      LOG_ERROR("'imageSize' is not specified.");
      exit(EXIT_FAILURE);
    }
  }

  {
    const auto ptr_maxSteps = GetValueHelpers::getScalarValue<int>("max_steps", *jsonValue);
    if (ptr_maxSteps != nullptr) {
      config.maxSteps = static_cast<int64_t>(*ptr_maxSteps);
    }
  }

  {
    const auto ptr_logEveryStep = GetValueHelpers::getScalarValue<int>("log_every_step", *jsonValue);
    if (ptr_logEveryStep != nullptr) {
      config.logEveryStep = static_cast<int64_t>(*ptr_logEveryStep);
    }
  }

  {
    const auto ptr_sampleEveryStep = GetValueHelpers::getScalarValue<int>("sample_every_step", *jsonValue);
    if (ptr_sampleEveryStep != nullptr) {
      config.sampleEveryStep = static_cast<int64_t>(*ptr_sampleEveryStep);
    }
  }

  {
    const auto ptr_nSamples = GetValueHelpers::getScalarValue<int>("num_samples", *jsonValue);
    if (ptr_nSamples != nullptr) {
      config.nSamples = static_cast<int64_t>(*ptr_nSamples);
    }
  }

  LOG_INFO("Done.");

  return config;
}

}  // namespace dmcpp::config
