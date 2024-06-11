#pragma once

#include <torch/torch.h>

#include <DiffusionModelC++/Model/Modules.hpp>

namespace dmcpp {
namespace model {

// ====================================================================================================
// ResConvBLock
// ====================================================================================================
struct ResConvBlockImpl : public ConditionedModuleImpl /*, public torch::nn::Cloneable<ResConvBlockImpl>*/ {
  ResConvBlockImpl(int64_t inFeatures,
                   int64_t inChannels,
                   int64_t midChannels,
                   int64_t outChannels,
                   int64_t nGroups = 32,
                   double dropoutRate = 0.0);

  torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx) override;

  void reset() override;

  AdaGN _norm0 = nullptr;
  torch::nn::GELU _act0 = nullptr;
  torch::nn::Conv2d _conv0 = nullptr;
  torch::nn::Dropout2d _dropout0 = nullptr;
  AdaGN _norm1 = nullptr;
  torch::nn::GELU _act1 = nullptr;
  torch::nn::Conv2d _conv1 = nullptr;
  torch::nn::Dropout2d _dropout1 = nullptr;

  torch::nn::Sequential _skipModules = nullptr;
};

TORCH_MODULE(ResConvBlock);

// ====================================================================================================
// DownBlock
// ====================================================================================================
struct DownBlockImpl : public ConditionedSequentialImpl {
  DownBlockImpl(int64_t nLayers,
                int64_t inFeatures,
                int64_t inChannels,
                int64_t midChannels,
                int64_t outChannels,
                int64_t nGroups = 32,
                int64_t headSize = 64,
                double dropoutRate = 0.0,
                bool downSample = false,
                bool selfAttention = false,
                bool crossAttention = false,
                int64_t encChannels = 0);
};

TORCH_MODULE(DownBlock);

// ====================================================================================================
// UpBlock
// ====================================================================================================
struct UpBlockImpl : public ConditionedSequentialImpl {
  UpBlockImpl(int64_t nLayers,
              int64_t inFeatures,
              int64_t inChannels,
              int64_t midChannels,
              int64_t outChannels,
              int64_t nGroups = 32,
              int64_t headSize = 64,
              double dropoutRate = 0.0,
              bool upSample = false,
              bool selfAttention = false,
              bool crossAttention = false,
              int64_t encChannels = 0);

  torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx) override;
  torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx, torch::Tensor& skip);
};

TORCH_MODULE(UpBlock);

// ====================================================================================================
// MappingNet
// ====================================================================================================
struct MappingNetImpl : public torch::nn::Cloneable<MappingNetImpl> {
  MappingNetImpl(int64_t inFeatures, int64_t outFeatures, int64_t nLayers = 2);

  torch::Tensor forward(torch::Tensor& x);

  void reset() override;

  torch::nn::Sequential _modules = nullptr;
};

TORCH_MODULE(MappingNet);

// ====================================================================================================
// UNet
// ====================================================================================================
struct UNetImpl : public torch::nn::Cloneable<UNetImpl> {
  UNetImpl(const std::vector<DownBlock>& downBlocks, const std::vector<UpBlock>& upBlocks);

  torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx);

  void reset() override;

  torch::nn::ModuleList _downBlocks = nullptr;
  torch::nn::ModuleList _upBlocks = nullptr;
};

TORCH_MODULE(UNet);

// ====================================================================================================
// ImageUNetModel
// ====================================================================================================
struct ImageUNetModelForwardArgs {
  ImageUNetModelForwardArgs()
      : mappingCond(),
        unetCond(),
        crossCond(),
        crossCondPadding(),
        returnVariance(false) {};

  torch::Tensor mappingCond;
  torch::Tensor unetCond;
  torch::Tensor crossCond;
  torch::Tensor crossCondPadding;
  bool returnVariance = false;
};

struct ImageUNetModelForwardReturn {
  ImageUNetModelForwardReturn()
      : output(),
        logVar() {};

  torch::Tensor output;
  torch::Tensor logVar;
};

struct ImageUNetModelImpl : public torch::nn::Cloneable<ImageUNetModelImpl> {
  ImageUNetModelImpl(int64_t inChannels,
                     int64_t inFeatures,
                     const std::vector<int64_t>& depth,
                     const std::vector<int64_t>& channels,
                     const std::vector<bool>& selfAttenDepth,
                     const std::vector<bool>& crossAttenDepth,
                     int64_t mappingCondDim = 0,
                     int64_t unetCondDim = 0,
                     int64_t crossCondDim = 0,
                     double dropoutRate = 0.0,
                     bool hasVariance = false);

  ImageUNetModelForwardReturn forward(const torch::Tensor& input,
                                      const torch::Tensor& sigma,
                                      const ImageUNetModelForwardArgs& args);

  void reset() override;

  bool _hasVariance;

  FourierFeatures _timestepEmbed = nullptr;
  torch::nn::Linear _mappingCond = nullptr;
  MappingNet _mapping = nullptr;
  torch::nn::Conv2d _inProj = nullptr;
  torch::nn::Conv2d _outProj = nullptr;
  UNet _uNet = nullptr;
};

TORCH_MODULE(ImageUNetModel);

}  // namespace model
}  // namespace dmcpp
