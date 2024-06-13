#include <DiffusionModelC++/Model/Modules.hpp>
#include <utility>

namespace dmcpp::model {

// ====================================================================================================
// ResidualBlock
// ====================================================================================================
torch::Tensor ResidualBlockImpl::forward(torch::Tensor& x, dmcpp::model::ConditionContext& conditionCtx) {
  return _mainModule->forward(x, conditionCtx) + _skipModule->forward(x);
}

void ResidualBlockImpl::reset() {
  _skipModule->reset();
  _mainModule->reset();
}

// ====================================================================================================
// ConditionedSequential
// ====================================================================================================
ConditionedSequentialImpl::ConditionedSequentialImpl()
    : _modules() {
}

ConditionedSequentialImpl::ConditionedSequentialImpl(const std::vector<std::shared_ptr<ConditionedModuleImpl>>& modules)
    : ConditionedSequentialImpl() {
  _modules = modules;
}

torch::Tensor ConditionedSequentialImpl::forward(torch::Tensor& x, dmcpp::model::ConditionContext& conditionCtx) {
  for (auto& module : _modules) {
    x = module->forward(x, conditionCtx);
  }

  return x;
}

void ConditionedSequentialImpl::reset() {
  for (auto& module : _modules) {
    module->reset();
  }
}

void ConditionedSequentialImpl::push_buck(std::string name, const std::shared_ptr<ConditionedModuleImpl>& module) {
  auto modulePtr = register_module(std::move(name), module);
  _modules.push_back(modulePtr);
}

size_t ConditionedSequentialImpl::size() const {
  return _modules.size();
}

// ====================================================================================================
// AdaGN
// ====================================================================================================
AdaGNImpl::AdaGNImpl(int64_t inFeatures,
                     int64_t outFeatures,
                     int64_t nGroups,
                     float epsilon)
    : _nGroups(nGroups),
      _epsilon(epsilon),
      _mapper(torch::nn::Linear(inFeatures, 2 * outFeatures)) {
  torch::nn::init::zeros_(_mapper->weight);
  torch::nn::init::zeros_(_mapper->bias);

  register_module("mapper", _mapper);
}

torch::Tensor AdaGNImpl::forward(torch::Tensor& x, torch::Tensor& condition) {
  // std::cout << "## AdaGNImpl::forward" << std::endl;
  // std::cout << "    x.size()      = " << x.sizes() << std::endl;

  auto chunks = _mapper(condition).chunk(2, 1);

  torch::Tensor weight = chunks[0].unsqueeze(-1).unsqueeze(-1);
  torch::Tensor bias = chunks[1].unsqueeze(-1).unsqueeze(-1);
  // std::cout << "    weight.size() = " << weight.sizes() << std::endl;
  // std::cout << "    bias.size()   = " << bias.sizes() << std::endl;

  x = torch::nn::functional::group_norm(x, torch::nn::functional::GroupNormFuncOptions(_nGroups).eps(_epsilon));
  // std::cout << "    x.size()      = " << x.sizes() << std::endl;

  return torch::addcmul(bias, x, weight + 1.0);
}

void AdaGNImpl::reset() {
  _mapper->reset();
}

// ====================================================================================================
// SelfAttention2D
// ====================================================================================================
SelfAttention2DImpl::SelfAttention2DImpl(int64_t inChannels,
                                         int64_t nHeads,
                                         AdaGN norm,
                                         float dropoutRate)
    : _nHeads(nHeads),
      _dropoutRate(dropoutRate),
      _norm(std::move(norm)) {
  _qkvProj = torch::nn::Conv2d(inChannels, inChannels * 3LL, 1);
  _outProj = torch::nn::Conv2d(inChannels, inChannels, 1);

  torch::nn::init::zeros_(_outProj->weight);
  torch::nn::init::zeros_(_outProj->bias);

  register_module("qkvProj", _qkvProj);
  register_module("outProj", _outProj);
  register_module("norm", _norm);
}

torch::Tensor SelfAttention2DImpl::forward(torch::Tensor& x, dmcpp::model::ConditionContext& conditionCtx) {
  // std::cout << "## SelfAttention2DImpl::forward" << std::endl;

  const int64_t b = x.size(0);
  const int64_t c = x.size(1);
  const int64_t h = x.size(2);
  const int64_t w = x.size(3);

  // std::cout << "    x.size()     = " << x.sizes() << std::endl;

  torch::Tensor qkv = _qkvProj->forward(_norm->forward(x, conditionCtx.condition));
  qkv = qkv.view({b, _nHeads * 3LL, c / _nHeads, h * w}).transpose(2, 3);
  // std::cout << "    qkv.size()   = " << qkv.sizes() << std::endl;

  torch::Tensor query = qkv.index({at::indexing::Slice(),
                                   at::indexing::Slice(0, _nHeads),
                                   at::indexing::Slice(),
                                   at::indexing::Slice()});
  torch::Tensor key = qkv.index({at::indexing::Slice(),
                                 at::indexing::Slice(_nHeads, 2LL * _nHeads),
                                 at::indexing::Slice(),
                                 at::indexing::Slice()});
  torch::Tensor value = qkv.index({at::indexing::Slice(),
                                   at::indexing::Slice(2LL * _nHeads, 3LL * _nHeads),
                                   at::indexing::Slice(),
                                   at::indexing::Slice()});
  // std::cout << "    query.size() = " << query.sizes() << std::endl;
  // std::cout << "    key.size()   = " << key.sizes() << std::endl;
  // std::cout << "    value.size() = " << value.sizes() << std::endl;

  torch::Tensor y = torch::scaled_dot_product_attention(query, key, value, {}, _dropoutRate);

  // std::cout << "    y.size()     = " << y.sizes() << std::endl;
  y = y.transpose(2, 3).contiguous().view({b, c, h, w});
  // std::cout << "    y.size()     = " << y.sizes() << std::endl;

  return x + _outProj(y);
}

void SelfAttention2DImpl::reset() {
  _norm->reset();
  _qkvProj->reset();
  _outProj->reset();
}

// ====================================================================================================
// CrossAttention2D
// ====================================================================================================
CrossAttention2DImpl::CrossAttention2DImpl(int64_t c_dec,
                                           int64_t c_enc,
                                           int64_t nHead,
                                           AdaGN normDec,
                                           float dropoutRate)
    : _nHeads(nHead),
      _dropoutRate(dropoutRate),
      _normDec(std::move(normDec)) {
  _normEnc = torch::nn::LayerNorm(torch::nn::LayerNormOptions({c_enc}));
  _qProj = torch::nn::Conv2d(c_dec, c_dec, 1);
  _kvProj = torch::nn::Linear(c_enc, c_dec * 2LL);
  _outProj = torch::nn::Conv2d(c_dec, c_dec, 1);

  torch::nn::init::zeros_(_outProj->weight);
  torch::nn::init::zeros_(_outProj->bias);

  register_module("qProj", _qProj);
  register_module("kvProj", _kvProj);
  register_module("normDec", _normDec);
  register_module("normDec", _normDec);
  register_module("outProj", _outProj);
}

torch::Tensor CrossAttention2DImpl::forward(torch::Tensor& x, ConditionContext& conditionCtx) {
  const int64_t b = x.size(0);
  const int64_t c = x.size(1);
  const int64_t h = x.size(2);
  const int64_t w = x.size(3);

  torch::Tensor query = _qProj->forward(_normDec->forward(x, conditionCtx.condition));
  query = query.view({b, _nHeads, c / _nHeads, h * w}).transpose(2, 3);

  torch::Tensor kv = _kvProj->forward(_normEnc->forward(conditionCtx.cross));
  kv = kv.view({b, -1, _nHeads * 2LL, c / _nHeads}).transpose(1, 2);

  torch::Tensor key = kv.index({at::indexing::Slice(),
                                at::indexing::Slice(0, _nHeads),
                                at::indexing::Slice(),
                                at::indexing::Slice()});
  torch::Tensor value = kv.index({at::indexing::Slice(),
                                  at::indexing::Slice(_nHeads, 2LL * _nHeads),
                                  at::indexing::Slice(),
                                  at::indexing::Slice()});

  torch::Tensor attentionMask = conditionCtx.crossPadding.unsqueeze(1).unsqueeze(2) * 10000.0;

  torch::Tensor y = torch::scaled_dot_product_attention(query, key, value, attentionMask, _dropoutRate);
  y = y.transpose(2, 3).contiguous().view({b, c, h, w});

  return x + _outProj(y);
}

void CrossAttention2DImpl::reset() {
  _normEnc->reset();
  _normDec->reset();
  _qProj->reset();
  _kvProj->reset();
  _outProj->reset();
}

// ====================================================================================================
// Downsample2D
// ====================================================================================================
Downsample2DImpl::Downsample2DImpl(torch::nn::functional::InterpolateFuncOptions::mode_t interp)
    : _interp(interp) {}

torch::Tensor Downsample2DImpl::forward(torch::Tensor& x, ConditionContext& conditionCtx) {
  return torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions()
                                                   .mode(_interp)
                                                   .scale_factor(SCALE_FACTOR)
                                                   .align_corners(true)
                                                   .recompute_scale_factor(false));
}

void Downsample2DImpl::reset() {
  // DO nothing;
}

// ====================================================================================================
// Upsample2D
// ====================================================================================================
Upsample2DImpl::Upsample2DImpl(torch::nn::functional::InterpolateFuncOptions::mode_t interp)
    : _interp(interp) {}

torch::Tensor Upsample2DImpl::forward(torch::Tensor& x, ConditionContext& conditionCtx) {
  return torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions()
                                                   .mode(_interp)
                                                   .scale_factor(SCALE_FACTOR)
                                                   .align_corners(true)
                                                   .recompute_scale_factor(false));
}

void Upsample2DImpl::reset() {
  // DO nothing;
}

// ====================================================================================================
// FourierFeatures
// ====================================================================================================
FourierFeaturesImpl::FourierFeaturesImpl(int64_t inFeatures,
                                         int64_t outFeatures,
                                         float std) {
  TORCH_CHECK(outFeatures % 2 == 0, "outFeatures must be even");

  _weights = torch::randn({outFeatures / 2LL, inFeatures}) * std;

  register_parameter("weight", _weights);
}

torch::Tensor FourierFeaturesImpl::forward(at::Tensor x) const {
  x = 2.0 * M_PI * x.matmul(_weights.t());

  return torch::cat({torch::cos(x), torch::sin(x)}, -1);
}

void FourierFeaturesImpl::reset() {
  _weights.reset();
}

}  // namespace dmcpp::model
