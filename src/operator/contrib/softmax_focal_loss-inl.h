// Modified from Official Caffe2 implementation
// Author: ddlee, me@ddlee.cn

// A multiclass form of Focal Loss designed for use in RetinaNet-like models.
// The input is assumed to be unnormalized scores (sometimes called 'logits')
// arranged in a 4D tensor with shape (N, C, H, W), where N is the number of
// elements in the batch, H and W are the height and width, and C = num_anchors *
// num_classes. The softmax is applied num_anchors times along the C axis.

// The softmax version of focal loss is:

//   FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t),

// where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
// s_j is the unnormalized score for class j.

// See: https://arxiv.org/abs/1708.02002 for details.

#ifndef MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_
#define MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include "../mshadow_op.h"
#include "../operator_common.h"

// namespace
namespace mxnet
{
namespace op
{

namespace focalloss
{
enum SoftmaxFocalLossOpInputs
{
  kData,
  kLabel,
  kNorm
};
enum SoftmaxFocalLossOpOutputs
{
  kLoss,
  kProb
};
enum SoftmaxFocalLossOpResource 
{ kTempSpace }; // mimicking protected losses_ buff_, need shapecheck
} // namespace focalloss

struct SoftmaxFocalLossParam : public dmlc::Parameter<SoftmaxFocalLossParam>
{
  float grad_scale;
  float alpha;
  float gamma;
  int num_classes;
  DMLC_DECLARE_PARAMETER(SoftmaxFocalLossParam)
  {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f).describe("(float) default 1.0; multiply the loss by this scale factor.");
    DMLC_DECLARE_FIELD(alpha).set_default(0.25f).describe("(float) default 0.25; Focal Loss's alpha hyper-parameter.");
    DMLC_DECLARE_FIELD(gamma).set_default(1.0f).describe("(float) default 1.0; Focal Loss's gamma hyper-parameter.");
    DMLC_DECLARE_FIELD(num_classes).set_default(81).describe("(int) default 81; number of classes in each softmax group.");
  }
};

template <typename xpu, typename DType>
class SoftmaxFocalLossOp : public Operator
{
public:
  explicit SoftmaxFocalLossOp(SoftmaxFocalLossParam p)
  {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args)
  {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);

    //TODO: shape check
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[focalloss::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> label = in_data[focalloss::kLabel].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> normalizer = in_data[focalloss::kNorm].get<xpu, 1, DType>(s);
    Tensor<xpu, 4, DType> loss = out_data[focalloss::kLoss].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> prob = out_data[focalloss::kProb].get<xpu, 4, DType>(s);
  
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(prob.CheckContiguous(), true);

    SoftmaxFocalLossForward(data, label, normalizer, loss, prob, param_.num_classes, param_.gamma, param_.alpha);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args)
  {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);

    // TODO: shape check

    Stream<xpu> *s = ctx.get_stream<xpu>();


    Tensor<xpu, 4, DType> data = in_data[focalloss::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> label = in_data[focalloss::kLabel].get<xpu, 4, DType>(s);
    Tensor<xpu, 1, DType> normalizer = in_data[focalloss::kNorm].get<xpu, 1, DType>(s);
    Tensor<xpu, 4, DType> prob = out_data[focalloss::kProb].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_in = in_grad[focalloss::kData].get<xpu, 4, DType>(s); //dX
    Tensor<xpu, 4, DType> grad_out = out_grad[focalloss::kLoss].get<xpu, 4, DType>(s); //dloss
    //Tensor<xpu, 4, DType> buff_ = aux_states[focalloss::kBuff].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> buff_ = ctx.requested[focalloss::kTempSpace]
      .get_space_typed<xpu, 4, DType>(label.shape_, s);
    
    buff_ = -1.0f;

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(prob.CheckContiguous(), true);

    SoftmaxFocalLossBackwardAcc(data, label, normalizer, prob, grad_in, grad_out, buff_, param_.num_classes, param_.gamma, param_.alpha);
  }

private:
  SoftmaxFocalLossParam param_;
}; // class SoftmaxFocalLossOp

// Decalre Factory function, used for dispatch specialization
template <typename xpu>
Operator *CreateOp(SoftmaxFocalLossParam param, int dtype);

#if DMLC_USE_CXX11
class SoftmaxFocalLossProp : public OperatorProperty
{
public:
  std::vector<std::string> ListArguments() const override
  {
    return {"data", "label", "normalizer"};
  }

  std::vector<std::string> ListOutputs() const override
  {
    return {"loss", "prob"};
  }

  int NumOutputs() const override
  {
    return 2;
  }

  int NumVisibleOutputs() const override
  {
    return 2;
  }

  void Init(const std::vector<std::pair<std::string, std::string>> &kwargs) override
  {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override
  {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override
  {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, label, normalizer]";

    // data: (N, C, H, W) C = num_anchors * num_class
    TShape dshape = in_shape->at(focalloss::kData);
    CHECK_EQ(dshape.ndim(), 4U) << "data should be a 4D tensor";

    // label: (N, num_anchors, H, W)
    TShape lshape = in_shape->at(focalloss::kLabel);
    CHECK_EQ(lshape.ndim(), 4U) << "label should be a 4D tensor";

    // normalizer: scalar
    TShape nshape = in_shape->at(focalloss::kNorm);
    CHECK_EQ(nshape.ndim(), 1U) << "Normalizer should be scalar";

    out_shape->clear();
    // loss: (N, num_anchors, H, W)
    out_shape->push_back(Shape4(dshape[0], lshape[1], dshape[2], dshape[3]));
    // prob: (N, C, H, W)
    out_shape->push_back(Shape4(dshape[0], dshape[1], dshape[2], dshape[3]));

    // aux_shape->clear();
    // // buff_: (N, num_anchors, H, W)
    // aux_shape->push_back(Shape4(dshape[0], lshape[1], dshape[2], dshape[3]));
    return true;
  }

  // bool InferType(std::vector<int> *in_type,
  //                std::vector<int> *out_type,
  //                std::vector<int> *aux_type) const override
  // {
  //   CHECK_EQ(in_type->size(), 2U);
  //   int dtype = (*in_type)[0];
  //   CHECK_EQ(dtype, (*in_type)[1]);
  //   CHECK_NE(dtype, -1) << "Input must have specified type";

  //   out_type->clear();
  //   out_type->push_back(dtype);
  //   out_type->push_back(dtype);

  //   aux_type->clear();
  //   aux_type->push_back(dtype);
  //   aux_type->push_back(dtype);
  //   return true;
  // }

  OperatorProperty *Copy() const override
  {
    SoftmaxFocalLossProp *softmax_focalloss_sym = new SoftmaxFocalLossProp();
    softmax_focalloss_sym->param_ = this->param_;
    return softmax_focalloss_sym;
  }

  std::string TypeString() const override
  {
    return "_contrib_SoftmaxFocalLoss";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override
  {
    return {out_grad[focalloss::kLoss], out_data[focalloss::kProb], in_data[focalloss::kData], in_data[focalloss::kLabel], };
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator *CreateOperator(Context ctx) const override
  {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

private:
  SoftmaxFocalLossParam param_;
}; // class SoftmaxFocalLossProp
#endif
} // namespace op
} // namespace mxnet
#endif // MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_
