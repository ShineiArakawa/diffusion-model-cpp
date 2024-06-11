#pragma once

#include <torch/torch.h>

#include <DiffusionModelC++/Util/Logging.hpp>
#include <vector>

namespace dmcpp {
namespace model {

struct ConditionContext {
  ConditionContext()
      : condition(),
        cross(),
        crossPadding() {};

  torch::Tensor condition;
  torch::Tensor cross;
  torch::Tensor crossPadding;
};

// ====================================================================================================
// ResidualBlock
// ====================================================================================================
struct ResidualBlockImpl : public torch::nn::Cloneable<ResidualBlockImpl> {
  torch::Tensor forward(torch::Tensor& x,
                        ConditionContext& conditionCtx);

  void reset() override;

  torch::nn::Sequential _skipModule = nullptr;
  torch::nn::Sequential _mainModule = nullptr;
};

TORCH_MODULE(ResidualBlock);

// ====================================================================================================
// ConditionedModule
// ====================================================================================================
struct ConditionedModuleImpl : public torch::nn::Cloneable<ConditionedModuleImpl> {
  virtual torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx) {
    LOG_ERROR("Not implemented!");
    exit(EXIT_FAILURE);
    return x;
  };

  virtual void reset() {
    LOG_ERROR("Not implemented!");
    exit(EXIT_FAILURE);
  };
};

// ====================================================================================================
// ConditionedSequential
// ====================================================================================================
struct ConditionedSequentialImpl : public torch::nn::Cloneable<ConditionedSequentialImpl> {
  ConditionedSequentialImpl();
  explicit ConditionedSequentialImpl(const std::vector<std::shared_ptr<ConditionedModuleImpl>>& modules);

  virtual torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx);

  void reset() override;

  void push_buck(std::string name, const std::shared_ptr<ConditionedModuleImpl>& module);

  size_t size() const;

  std::vector<std::shared_ptr<ConditionedModuleImpl>> _modules;
};

TORCH_MODULE(ConditionedSequential);

// ====================================================================================================
// AdaGN
// ====================================================================================================
struct AdaGNImpl : public torch::nn::Cloneable<AdaGNImpl> {
  AdaGNImpl(int64_t inFeatures,
            int64_t outFeatures,
            int64_t nGroups,
            float epsilon = 1e-5);

  torch::Tensor forward(torch::Tensor& x, torch::Tensor& condition);

  void reset() override;

  int64_t _nGroups;
  float _epsilon;

  torch::nn::Linear _mapper = nullptr;
};

TORCH_MODULE(AdaGN);

// ====================================================================================================
// SelfAttention2D
// ====================================================================================================
struct SelfAttention2DImpl : public ConditionedModuleImpl /*, public torch::nn::Cloneable<SelfAttention2DImpl>*/ {
  SelfAttention2DImpl(int64_t inChannels,
                      int64_t nHeads,
                      AdaGN norm,
                      float dropoutRate = 0.0f);

  torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx) override;

  void reset() override;

  int64_t _nHeads;
  float _dropoutRate;
  AdaGN _norm = nullptr;
  torch::nn::Conv2d _qkvProj = nullptr;
  torch::nn::Conv2d _outProj = nullptr;
};

TORCH_MODULE(SelfAttention2D);

// ====================================================================================================
// CrossAttention2D
// ====================================================================================================
struct CrossAttention2DImpl : public ConditionedModuleImpl /*, public torch::nn::Cloneable<CrossAttention2DImpl>*/ {
  CrossAttention2DImpl(int64_t c_dec,
                       int64_t c_enc,
                       int64_t nHeads,
                       AdaGN normDec,
                       float dropoutRate = 0.0f);

  torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx) override;

  void reset() override;

  int64_t _nHeads;
  float _dropoutRate;

  torch::nn::LayerNorm _normEnc = nullptr;
  AdaGN _normDec = nullptr;
  torch::nn::Conv2d _qProj = nullptr;
  torch::nn::Linear _kvProj = nullptr;
  torch::nn::Conv2d _outProj = nullptr;
};

TORCH_MODULE(CrossAttention2D);

// ====================================================================================================
// Downsample2D
// ====================================================================================================
struct Downsample2DImpl : public ConditionedModuleImpl /*, public torch::nn::Cloneable<Downsample2DImpl>*/ {
  inline static const std::vector<double> SCALE_FACTOR = {0.5, 0.5};

  explicit Downsample2DImpl(torch::nn::functional::InterpolateFuncOptions::mode_t interp = torch::kBilinear);

  torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx) override;

  void reset() override;

  torch::nn::functional::InterpolateFuncOptions::mode_t _interp;
};

TORCH_MODULE(Downsample2D);

// ====================================================================================================
// Upsample2D
// ====================================================================================================
struct Upsample2DImpl : public ConditionedModuleImpl /*, public torch::nn::Cloneable<Upsample2DImpl>*/ {
  inline static const std::vector<double> SCALE_FACTOR = {2.0, 2.0};

  explicit Upsample2DImpl(torch::nn::functional::InterpolateFuncOptions::mode_t interp = torch::kBilinear);

  torch::Tensor forward(torch::Tensor& x, ConditionContext& conditionCtx) override;

  void reset() override;

  torch::nn::functional::InterpolateFuncOptions::mode_t _interp;
};

TORCH_MODULE(Upsample2D);

// ====================================================================================================
// FourierFeatures
// ====================================================================================================
struct FourierFeaturesImpl : public torch::nn::Cloneable<FourierFeaturesImpl> {
  FourierFeaturesImpl(int64_t inFeatures, int64_t outFeatures, float std = 1.0f);

  torch::Tensor forward(at::Tensor x) const;

  void reset() override;

  torch::Tensor _weights;
};

TORCH_MODULE(FourierFeatures);

}  // namespace model
}  // namespace dmcpp