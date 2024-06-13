#include <DiffusionModelC++/Model/Model.hpp>
#include <algorithm>

// #define DEBUG_DMCPP_MODEL

#ifdef DEBUG_DMCPP_MODEL
#include <DiffusionModelC++/Util/ImageUtil.hpp>
#endif

namespace dmcpp::model {
// ====================================================================================================
// ResConvBLock
// ====================================================================================================
ResConvBlockImpl::ResConvBlockImpl(int64_t inFeatures,
                                   int64_t inChannels,
                                   int64_t midChannels,
                                   int64_t outChannels,
                                   int64_t nGroups,
                                   double dropoutRate)
    : _skipModules() {
  int64_t nGroupsValidIn = inChannels / nGroups;
  if (nGroupsValidIn < 1) {
    nGroupsValidIn = 1;
  }

  int64_t nGroupsValidMid = midChannels / nGroups;
  if (nGroupsValidMid < 1) {
    nGroupsValidMid = 1;
  }

  // Main modules
  _norm0 = AdaGN(inFeatures, inChannels, nGroupsValidIn);
  _act0 = torch::nn::GELU();
  _conv0 = torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, midChannels, 3).padding(1));
  _dropout0 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropoutRate).inplace(true));
  _norm1 = AdaGN(inFeatures, midChannels, nGroupsValidMid);
  _act1 = torch::nn::GELU();
  _conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(midChannels, outChannels, 3).padding(1));
  _dropout1 = torch::nn::Dropout2d(torch::nn::Dropout2dOptions(dropoutRate).inplace(true));

  torch::nn::init::zeros_(_conv1->weight);
  torch::nn::init::zeros_(_conv1->bias);

  // Skip module
  if (inChannels == outChannels) {
    _skipModules->push_back(torch::nn::Identity());
  } else {
    auto skipConv = torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels, outChannels, 1).bias(false));
    torch::nn::init::orthogonal_(skipConv->weight);
    _skipModules->push_back(skipConv);
  }

  // Register
  register_module("norm0", _norm0);
  register_module("act0", _act0);
  register_module("conv0", _conv0);
  register_module("dropout0", _dropout0);
  register_module("norm1", _norm1);
  register_module("act1", _act1);
  register_module("conv1", _conv1);
  register_module("dropout1", _dropout1);
  register_module("skipModules", _skipModules);
}

torch::Tensor ResConvBlockImpl::forward(torch::Tensor& x, ConditionContext& conditionCtx) {
  // std::cout << "ResConvBlockImpl::forward" << std::endl;

  // std::cout << "    x.size() = " << x.sizes() << std::endl;
  torch::Tensor y = _norm0->forward(x, conditionCtx.condition);
  // std::cout << "    y.size() = " << y.sizes() << std::endl;
  y = _act0->forward(y);
  // std::cout << "    y.size() = " << y.sizes() << std::endl;
  y = _conv0->forward(y);
  // std::cout << "    y.size() = " << y.sizes() << std::endl;
  y = _dropout0->forward(y);
  // std::cout << "    y.size() = " << y.sizes() << std::endl;
  y = _norm1->forward(y, conditionCtx.condition);
  // std::cout << "    y.size() = " << y.sizes() << std::endl;
  y = _act1->forward(y);
  // std::cout << "    y.size() = " << y.sizes() << std::endl;
  y = _conv1->forward(y);
  // std::cout << "    y.size() = " << y.sizes() << std::endl;
  y = _dropout1->forward(y);
  // std::cout << "    y.size() = " << y.sizes() << std::endl;

  return y + _skipModules->forward(x);
}

void ResConvBlockImpl::reset() {
  _norm0->reset();
  _act0->reset();
  _conv0->reset();
  _dropout0->reset();
  _norm1->reset();
  _act1->reset();
  _conv1->reset();
  _dropout1->reset();
}

// ====================================================================================================
// DownBlock
// ====================================================================================================
DownBlockImpl::DownBlockImpl(int64_t nLayers,
                             int64_t inFeatures,
                             int64_t inChannels,
                             int64_t midChannels,
                             int64_t outChannels,
                             int64_t nGroups,
                             int64_t headSize,
                             double dropoutRate,
                             bool downSample,
                             bool selfAttention,
                             bool crossAttention,
                             int64_t encChannels)
    : ConditionedSequentialImpl() {
  if (downSample) {
    push_buck("downSample", std::make_shared<Downsample2DImpl>());
  }

  for (int64_t iLayer = 0; iLayer < nLayers; ++iLayer) {
    const int64_t tmpInChannels = iLayer == 0LL ? inChannels : midChannels;
    const int64_t tmpOutChannels = iLayer < nLayers - 1LL ? midChannels : outChannels;

    push_buck("resConvBlock" + std::to_string(iLayer),
              std::make_shared<ResConvBlockImpl>(inFeatures,
                                                 tmpInChannels,
                                                 midChannels,
                                                 tmpOutChannels,
                                                 nGroups,
                                                 dropoutRate));

    // Attention heads
    int64_t nHeads = tmpOutChannels / headSize;
    if (nHeads < 1LL) {
      nHeads = 1LL;
    }

    // Group size
    int64_t nGroupsValid = tmpOutChannels / nGroups;
    if (nGroupsValid < 1LL) {
      nGroupsValid = 1LL;
    }

    if (selfAttention) {
      AdaGN normModule(inFeatures, tmpOutChannels, nGroupsValid);
      push_buck("selfAttention" + std::to_string(iLayer),
                std::make_shared<SelfAttention2DImpl>(tmpOutChannels, nHeads, normModule, dropoutRate));
    }

    if (crossAttention) {
      AdaGN normModule(inFeatures, tmpOutChannels, nGroupsValid);
      push_buck("crossAttention" + std::to_string(iLayer),
                std::make_shared<CrossAttention2DImpl>(tmpOutChannels, encChannels, nHeads, normModule, dropoutRate));
    }
  }
}

// ====================================================================================================
// UpBlock
// ====================================================================================================
UpBlockImpl::UpBlockImpl(int64_t nLayers,
                         int64_t inFeatures,
                         int64_t inChannels,
                         int64_t midChannels,
                         int64_t outChannels,
                         int64_t nGroups,
                         int64_t headSize,
                         double dropoutRate,
                         bool upSample,
                         bool selfAttention,
                         bool crossAttention,
                         int64_t encChannels)
    : ConditionedSequentialImpl() {
  for (int64_t iLayer = 0; iLayer < nLayers; ++iLayer) {
    const int64_t tmpInChannels = iLayer == 0LL ? inChannels : midChannels;
    const int64_t tmpOutChannels = iLayer < nLayers - 1LL ? midChannels : outChannels;

    push_buck("resConvBlock" + std::to_string(iLayer),
              std::make_shared<ResConvBlockImpl>(inFeatures,
                                                 tmpInChannels,
                                                 midChannels,
                                                 tmpOutChannels,
                                                 nGroups,
                                                 dropoutRate));

    // Attention heads
    int64_t nHeads = tmpOutChannels / headSize;
    if (nHeads < 1LL) {
      nHeads = 1LL;
    }

    // Group size
    int64_t nGroupsValid = tmpOutChannels / nGroups;
    if (nGroupsValid < 1LL) {
      nGroupsValid = 1LL;
    }

    if (selfAttention) {
      AdaGN normModule(inFeatures, tmpOutChannels, nGroupsValid);
      push_buck("selfAttention" + std::to_string(iLayer),
                std::make_shared<SelfAttention2DImpl>(tmpOutChannels, nHeads, normModule, dropoutRate));
    }

    if (crossAttention) {
      AdaGN normModule(inFeatures, tmpOutChannels, nGroupsValid);
      push_buck("crossAttention" + std::to_string(iLayer),
                std::make_shared<CrossAttention2DImpl>(tmpOutChannels, encChannels, nHeads, normModule, dropoutRate));
    }
  }

  if (upSample) {
    push_buck("upSample", std::make_shared<Upsample2DImpl>());
  }
}

torch::Tensor UpBlockImpl::forward(torch::Tensor& x, ConditionContext& conditionCtx) {
  return ConditionedSequentialImpl::forward(x, conditionCtx);
}

torch::Tensor UpBlockImpl::forward(torch::Tensor& x, ConditionContext& conditionCtx, torch::Tensor& skip) {
  x = torch::cat({x, skip}, 1);
  return ConditionedSequentialImpl::forward(x, conditionCtx);
}

// ====================================================================================================
// MappingNet
// ====================================================================================================
MappingNetImpl::MappingNetImpl(int64_t inFeatures, int64_t outFeatures, int64_t nLayers)
    : _modules() {
  for (int64_t iLayer = 0; iLayer < nLayers; ++iLayer) {
    auto linear = torch::nn::Linear(iLayer == 0LL ? inFeatures : outFeatures, outFeatures);
    torch::nn::init::orthogonal_(linear->weight);

    _modules->push_back(linear);
    _modules->push_back(torch::nn::GELU());
  }

  register_module("modules", _modules);
}

torch::Tensor MappingNetImpl::forward(torch::Tensor& x) {
  return _modules->forward(x);
}

void MappingNetImpl::reset() {
  _modules->reset();
}

// ====================================================================================================
// UNet
// ====================================================================================================
UNetImpl::UNetImpl(const std::vector<DownBlock>& downBlocks, const std::vector<UpBlock>& upBlocks)
    : _downBlocks(),
      _upBlocks() {
  for (auto& module : downBlocks) {
    _downBlocks->push_back(module);
  }

  for (auto& module : upBlocks) {
    _upBlocks->push_back(module);
  }

  register_module("downBlocks", _downBlocks);
  register_module("upBlocks", _upBlocks);
}

torch::Tensor UNetImpl::forward(torch::Tensor& x, ConditionContext& conditionCtx) {
  // std::cout << "UNetImpl::forward" << std::endl;

  std::vector<torch::Tensor> hidden;

  for (auto& module : *_downBlocks) {
    // std::cout << "[UNetImpl]    x.size() = " << x.sizes() << std::endl;
    x = module->as<DownBlock>()->forward(x, conditionCtx);
    // std::cout << "[UNetImpl]    x.size() = " << x.sizes() << std::endl;
    hidden.push_back(x);
  }

  std::reverse(hidden.begin(), hidden.end());

  const size_t nUpBlocks = _upBlocks->size();
  for (size_t iBlock = 0; iBlock < nUpBlocks; ++iBlock) {
    // std::cout << "[UNetImpl]    iBlock   = " << iBlock << std::endl;
    // std::cout << "[UNetImpl]    x.size() = " << x.sizes() << std::endl;
    if (iBlock == 0) {
      x = _upBlocks[iBlock]->as<UpBlock>()->forward(x, conditionCtx);
    } else {
      x = _upBlocks[iBlock]->as<UpBlock>()->forward(x, conditionCtx, hidden[iBlock]);
    }
    // std::cout << "[UNetImpl]    x.size() = " << x.sizes() << std::endl;
  }

  return x;
}

void UNetImpl::reset() {
  _downBlocks->reset();
  _upBlocks->reset();
}

// ====================================================================================================
// ImageUNetModel
// ====================================================================================================
ImageUNetModelImpl::ImageUNetModelImpl(int64_t inChannels,
                                       int64_t inFeatures,
                                       const std::vector<int64_t>& depth,
                                       const std::vector<int64_t>& channels,
                                       const std::vector<bool>& selfAttenDepth,
                                       const std::vector<bool>& crossAttenDepth,
                                       int64_t mappingCondDim,
                                       int64_t unetCondDim,
                                       int64_t crossCondDim,
                                       double dropoutRate,
                                       bool hasVariance)
    : _hasVariance(hasVariance) {
  {
    // Mapping network
    _timestepEmbed = FourierFeatures(1, inFeatures);

    if (mappingCondDim > 0) {
      _mappingCond = torch::nn::Linear(torch::nn::LinearOptions(mappingCondDim, inFeatures).bias(false));
      register_module("mappingCond", _mappingCond);
    }

    _mapping = MappingNet(inFeatures, inFeatures);
  }

  {
    // Projection layers
    const int64_t outChannels = hasVariance ? (inChannels + 1) : inChannels;
    _inProj = torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannels + unetCondDim, channels[0], 1));
    _outProj = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels[0], outChannels, 1));

    torch::nn::init::zeros_(_outProj->weight);
    torch::nn::init::zeros_(_outProj->bias);
  }

  {
    // UNet
    const auto nDepth = static_cast<int64_t>(depth.size());

    // Down blocks
    std::vector<DownBlock> downBlocks;

    for (int64_t iBlock = 0; iBlock < nDepth; ++iBlock) {
      int64_t index = iBlock - 1;

      if (index < 0) {
        index = 0;
      }

      const int64_t tmpInChannels = channels[index];

      downBlocks.emplace_back(depth[iBlock],
                              inFeatures,
                              tmpInChannels,
                              channels[iBlock],
                              channels[iBlock],
                              32,
                              64,
                              dropoutRate,
                              iBlock > 0,
                              selfAttenDepth[iBlock],
                              crossAttenDepth[iBlock],
                              crossCondDim);
    }

    // Up blocks
    std::vector<UpBlock> upBlocks;

    for (int64_t iBlock = 0; iBlock < nDepth; ++iBlock) {
      int64_t index = iBlock - 1;

      if (index < 0) {
        index = 0;
      }

      const int64_t tmpInChannels = iBlock < nDepth - 1LL ? 2LL * channels[iBlock] : channels[iBlock];
      const int64_t tmpOutChannels = channels[index];

      upBlocks.emplace_back(depth[iBlock],
                            inFeatures,
                            tmpInChannels,
                            channels[iBlock],
                            tmpOutChannels,
                            32,
                            64,
                            dropoutRate,
                            iBlock > 0,
                            selfAttenDepth[iBlock],
                            crossAttenDepth[iBlock],
                            crossCondDim);
    }

    // UNet
    std::reverse(upBlocks.begin(), upBlocks.end());

    _uNet = UNet(downBlocks, upBlocks);
  }

  register_module("timestepEmbed", _timestepEmbed);
  register_module("mapping", _mapping);
  register_module("inProj", _inProj);
  register_module("outProj", _outProj);
  register_module("uNet", _uNet);
}

ImageUNetModelForwardReturn ImageUNetModelImpl::forward(const torch::Tensor& input,
                                                        const torch::Tensor& sigma,
                                                        const ImageUNetModelForwardArgs& args) {
  const int64_t b = input.size(0);

  at::Tensor noiseCond = sigma.log() / 4.0;
  noiseCond = _timestepEmbed->forward(noiseCond.view({b, 1}));
  const at::Tensor& mappingCond = args.mappingCond.defined() ? _mappingCond(args.mappingCond) : torch::zeros_like(noiseCond);
  at::Tensor cond = noiseCond + mappingCond;
  const at::Tensor& mappedCond = _mapping->forward(cond);

  at::Tensor modelInput = input;
  modelInput = modelInput.contiguous();

  ConditionContext condCtx;
  condCtx.condition = mappedCond;

  if (args.unetCond.defined()) {
    modelInput = torch::cat({modelInput, args.unetCond}, 1);
  }

  if (args.crossCond.defined()) {
    condCtx.cross = args.crossCond;
    condCtx.crossPadding = args.crossCondPadding;
  }

#ifdef DEBUG_DMCPP_MODEL
  std::cout << "modelInput.is_contiguous() : " << (modelInput.is_contiguous() ? "true" : "false") << std::endl;
  std::cout << modelInput.sizes() << std::endl;
  util::DEBUG_saveImages(modelInput, "/home/araka/Projects/diffusion-model-cpp/debug/model/0");
#endif
  modelInput = _inProj->forward(modelInput);
#ifdef DEBUG_DMCPP_MODEL
  std::cout << modelInput.sizes() << std::endl;
  util::DEBUG_saveImages(modelInput, "/home/araka/Projects/diffusion-model-cpp/debug/model/1");
#endif
  modelInput = _uNet->forward(modelInput, condCtx);
#ifdef DEBUG_DMCPP_MODEL
  std::cout << modelInput.sizes() << std::endl;
  util::DEBUG_saveImages(modelInput, "/home/araka/Projects/diffusion-model-cpp/debug/model/2");
#endif
  modelInput = _outProj->forward(modelInput);
#ifdef DEBUG_DMCPP_MODEL
  std::cout << modelInput.sizes() << std::endl;
  util::DEBUG_saveImages(modelInput, "/home/araka/Projects/diffusion-model-cpp/debug/model/3");
#endif

  ImageUNetModelForwardReturn returnVars;

  if (_hasVariance) {
    const at::Tensor& output = modelInput.index({torch::indexing::Slice(),
                                                 torch::indexing::Slice(0, -1),
                                                 torch::indexing::Slice(),
                                                 torch::indexing::Slice()});
    const at::Tensor& logVar = modelInput.index({torch::indexing::Slice(),
                                                 -1,
                                                 torch::indexing::Slice(),
                                                 torch::indexing::Slice()})
                                   .flatten(1)
                                   .mean(1);

    returnVars.output = output;

    if (args.returnVariance) {
      returnVars.logVar = logVar;
    }
  } else {
    returnVars.output = modelInput;
  }

  return returnVars;
}

void ImageUNetModelImpl::reset() {
  // _timestepEmbed->reset();
  // _mapping->reset();
  // _inProj->reset();
  // _outProj->reset();
  // _uNet->reset();

  // if (!_mappingCond.is_empty()) {
  //   _mappingCond->reset();
  // }
}

}  // namespace dmcpp::model