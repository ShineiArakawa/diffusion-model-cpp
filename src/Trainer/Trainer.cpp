#include <DiffusionModelC++/Trainer/LRScheduler.hpp>
#include <DiffusionModelC++/Trainer/Trainer.hpp>
#include <DiffusionModelC++/Util/FileUtil.hpp>
#include <DiffusionModelC++/Util/ImageUtil.hpp>
#include <DiffusionModelC++/Util/Logging.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <utility>

namespace dmcpp::trainer {

Trainer::Trainer(const config::Config& config,
                 diffusion::KarrasDiffusion& model,
                 diffusion::KarrasDiffusion& emaModel)
    : _config(config),
      _model(model),
      _modelEMA(emaModel),
      _step() {
  // Create log dir
  LOG_INFO("Set log dir to" + _config.logDir);
  util::FileUtil::mkdirs(_config.logDir);

  // Set device
  if (config.deviceID >= 0) {
    _device = torch::Device(torch::kCUDA, config.deviceID);
  } else {
    _device = torch::Device(torch::kCPU);
  }

  // Move models to device
  _model->to(_device);
  _modelEMA->to(_device);

  // NOTE: Clone model to EMA model
  // _modelEMA = std::dynamic_pointer_cast<diffusion::KarrasDiffusionImpl>(_model->clone());

  // Set optimizer
  switch (config.optimizer.type) {
    case dmcpp::config::OptimizerType::ADAMW:
      _optimizer = std::make_shared<torch::optim::AdamW>(_model->parameters(),
                                                         torch::optim::AdamWOptions(config.optimizer.lr)
                                                             .betas({config.optimizer.betas[0], config.optimizer.betas[1]})
                                                             .eps(config.optimizer.eps)
                                                             .weight_decay(config.optimizer.weightDecay));
      break;
    default:
      LOG_CRITICAL("Invalid optimizer type");
      exit(EXIT_FAILURE);
  }

  // LR scheduler
  switch (config.lr_scheduler.type) {
    case dmcpp::config::LRSchedulerType::CONSTANT:
      _lrScheduler = std::make_shared<ConstantLRWithWarmup>(*_optimizer, config.lr_scheduler.warmup);
      break;
    default:
      LOG_CRITICAL("Invalid LR scheduler type");
      exit(EXIT_FAILURE);
  }

  // EMA scheduler
  _EMAScheduler = std::make_shared<EMAWarmup>(1.0, config.ema.power, 0.0, config.ema.maxValue);

  // Dataset
  auto dataset = ImageFolderDataset(config.dataset.root,
                                    config.imageSize,
                                    config.imageSize,
                                    config.dataset.extension);
  auto mappedDataset = dataset.map(torch::data::transforms::Stack<>());

  // DataLoader
  _dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(mappedDataset),
      torch::data::DataLoaderOptions()
          .batch_size(config.dataset.batchSize)
          .drop_last(true)
          .workers(config.dataset.numWorkers));

  // Sampler
  _sampler = std::make_shared<diffusion::CosineInterpolatedSampler>(config);
}

Trainer::~Trainer() = default;

void Trainer::fit() {
  LOG_INFO("Start training ...");

  auto startTime = std::chrono::high_resolution_clock::now();

  while (true) {
    for (auto& batch : *_dataLoader) {
      const torch::Tensor& image = batch.data.to(_device);
      const torch::Tensor& noise = torch::randn_like(image);

      const torch::Tensor& sigma = _sampler->sample({image.size(0)}, _device, torch::kFloat32);

      model::ImageUNetModelForwardArgs args;

      torch::Tensor loss = _model->loss(image, noise, sigma, args).mean();

      loss.backward();

      _optimizer->step();
      _lrScheduler->step();
      _optimizer->zero_grad();

      ++_step;

      // EMA update
      {
        torch::NoGradGuard no_grad;

        const double& emaDecay = _EMAScheduler->get_value();
        updateEMAModel(_model, _modelEMA, emaDecay);
      }

      if (_step % _config.logEveryStep == 0) {
        torch::NoGradGuard no_grad;

        auto currentTime = std::chrono::high_resolution_clock::now();
        double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count();

        LOG_INFO("Step " + std::to_string(_step) + " / " + std::to_string(_config.maxSteps) + " , Loss : " + std::to_string(loss.item<double>()) + " , Elapsed time : " + std::to_string(elapsedTime * 1e-6) + " [sec]");
      }

      if (_step % _config.sampleEveryStep == 0) {
        torch::NoGradGuard no_grad;
        LOG_INFO("Sampling ...");

        const torch::Tensor& x = torch::randn({_config.nSamples, _config.model.inChannels, _config.imageSize, _config.imageSize}, torch::TensorOptions(_device)) * _config.sampler.sigmaMax;
        const torch::Tensor& sigmas = diffusion::getSigmasKarras(50, _config.sampler.sigmaMin, _config.sampler.sigmaMax, 7.0, _device);
        const torch::Tensor& sampled = diffusion::sample_heun(_modelEMA, x, sigmas);

        const std::string sampleDirPath = util::FileUtil::join(util::FileUtil::join(_config.logDir, "sampled"), "step=" + std::to_string(_step));
        util::FileUtil::mkdirs(sampleDirPath);

        for (int64_t iImage = 0; iImage < _config.nSamples; ++iImage) {
          const std::string& filePath = util::FileUtil::join(sampleDirPath, "sample_" + std::to_string(iImage) + ".png");
          const cv::Mat& image = util::tensorToCv2Mat(sampled[iImage]);
          util::saveImage(image, filePath);
        }

        LOG_INFO("Done.");
      }

      if (_step % _config.checkpointEveryStep == 0) {
        torch::NoGradGuard no_grad;
        save();
      }

      if (_step == _config.maxSteps) {
        break;
      }
    }
  }
}

void Trainer::save() {
  torch::NoGradGuard no_grad;

  LOG_INFO("Saving checkpoint ...");

  torch::serialize::OutputArchive archive;

  {
    torch::serialize::OutputArchive tmpArchive;
    _model->save(tmpArchive);
    archive.write("model", tmpArchive);
  }

  {
    torch::serialize::OutputArchive tmpArchive;
    _modelEMA->save(tmpArchive);
    archive.write("ema_model", tmpArchive);
  }

  {
    torch::serialize::OutputArchive tmpArchive;
    _optimizer->save(tmpArchive);
    archive.write("optimizer", tmpArchive);
  }

  {
    torch::serialize::OutputArchive tmpArchive;
    const auto& stateDict = _EMAScheduler->state_dict();

    for (auto iter = stateDict.begin(); iter != stateDict.end(); ++iter) {
      tmpArchive.write(iter->first, iter->second);
    }

    archive.write("ema_sched", stateDict);
  }

  archive.write("step", _step);

  const std::string filePath = util::FileUtil::join(util::FileUtil::join(_config.logDir, "checkpoints"), "checkpoint_step=" + std::to_string(_step) + ".pth");
  util::FileUtil::mkdirs(util::FileUtil::dirPath(filePath));

  archive.save_to(filePath);

  LOG_INFO("Saving checkpoint to " + filePath);
}

}  // namespace dmcpp::trainer
