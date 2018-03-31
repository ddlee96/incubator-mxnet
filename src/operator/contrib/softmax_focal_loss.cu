// Modified from Official Caffe2 implementation
// Author: ddlee, me@ddlee.cn

#include "./softmax_focal_loss-inl.h"
#include <mshadow/tensor.h>
#include "../mshadow_op.h"

namespace mshadow {
namespace cuda {

    template<typename DType>
    __global__ void SpatialSoftmaxKernel(const int N, const int A,
        const int H, const int W, const DType *Xdata, DType *Pdata,
        const int num_classes) {
        CUDA_KERNEL_LOOP(index, N * A * H * W) {
        int D = num_classes * A;
        int x = index % W;
        int y = (index / W) % H;
        int a = (index / (W * H)) % A;
        int i = index / W / H / A;

        // Subtract max on each cell for numerical reasons
        float max_val = -FLT_MAX;
        for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
            int idx = i * (H * W * D) +  c * (H * W) + y * W + x;
            max_val = max(max_val, Xdata[idx]);
        }
        // Exponentiate
        float expsum = 0.0f;
        for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
            int idx = i * (H * W * D) + c * (H * W) + y * W + x;
            float expx = exp(Xdata[idx] - max_val);
            Pdata[idx] = expx;
            expsum += expx;
        }
        // Normalize
        for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
            int idx = i * (H * W * D) + c * (H * W) + y * W + x;
            Pdata[idx] /= expsum;
        }
        }
    }

    template<typename DType>
    __global__ void SoftmaxFocalLossKernel(
        const int N, const int A, const int H, const int W,
        const DType *Pdata, const DType *targets, DType *losses,
        const DType *weight_pos, const float gamma, const float alpha,
        const int num_classes) {
        CUDA_KERNEL_LOOP(i, N * A * H * W) {
        int D = A * num_classes;
        int x = i % W;
        int y = (i / W) % H;
        int a = (i / (W * H)) % A;
        int n = i / (W * H * A);
        const int label = static_cast<int>(targets[i]);

        float Np = max(weight_pos[0], 1.0);
        float z = (label == 0) * (1 - alpha) / Np +
                    (label >= 1) * alpha / Np;

        losses[i] = 0.0;
        if (label >= 0) {
            int offset = a * num_classes;
            int idx = n * (H * W * D) + (offset + label) * (H * W) + y * W + x;
            losses[i] =
                -(pow(1.0 - Pdata[idx], gamma) *
                log(max(Pdata[idx], FLT_MIN))) * z;
        }
        }
    }

    template<typename DType>
    __global__ void SoftmaxFocalLossGradientWeightKernel(
        const int N, const int A, const int H, const int W,
        const DType *Pdata, const DType *targets, DType *buff,
        const DType *weight_pos, const float gamma, const float alpha,
        const int num_classes) {
        CUDA_KERNEL_LOOP(i, N * A * H * W) {
        int D = A * num_classes;
        int x = i % W;
        int y = (i / W) % H;
        int a = (i / (W * H)) % A;
        int n = i / (W * H * A);
        const int label = static_cast<int>(targets[i]);
        float Np = max(weight_pos[0], 1.0);
        float z =  (label == 0) * (1 - alpha) / Np +
                    (label >= 1) * alpha / Np;

        buff[i] = 0.0;
        if (label >= 0) {
            int offset = a * num_classes;
            int idx = n * (H * W * D) + (offset + label) * (H * W) + y * W + x;
            float onemp = 1. - Pdata[idx];
            float p = Pdata[idx];
            buff[i] =
                (-pow(onemp, gamma) +
                gamma * pow(onemp, gamma - 1) * p * log(max(p, FLT_MIN))) * z;
        }
        }
    }

    template<typename DType>
    __global__ void SoftmaxFocalLossGradientKernel(
        const int N, const int D, const int H, const int W,
        const DType *Pdata, const DType *targets, const DType *buff,
        const DType *d_loss_data, DType *dX, const int num_classes) {
        CUDA_KERNEL_LOOP(i, N * D * H * W) {
        int A = D / num_classes;
        int x = i % W;
        int y = (i / W) % H;
        int d = (i / (W * H)) % D;
        int a = d / num_classes;
        int c = d % num_classes;
        int n = i / (W * H * D);
        float d_loss = *d_loss_data;

        int ind = n * (H * W * A) + a * (H * W) + y * W + x;
        const int label = static_cast<int>(targets[ind]);

        float c1 = (label >= 0) * 1.0;
        float c2 = (label == c) * 1.0;
        dX[i] = 0.0;
        dX[i] = c1 * d_loss * buff[ind] * (c2 - Pdata[i]);
        }
    }

    template<typename DType>
    inline void SoftmaxFocalLossForward(const Tensor<gpu, 4, DType> &X, // Logits; data
                                        const Tensor<gpu, 4, DType> &T, // Labels; labels
                                        const Tensor<gpu, 1, DType> &wp, // num of forground ; normalizer
                                        const Tensor<gpu, 1, DType> &avg_loss, // average loss as output; loss
                                        const Tensor<gpu, 4, DType> &P, //softmax probability, going to be re-used in gradient; prob
                                        const Tensor<gpu, 4, DType> &losses_, // aux losses_ Tensor
                                        const int num_classes_,
                                        const float gamma_,
                                        const float alpha_)
    {
    int N = X.size(0);
    int D = X.size(1);
    int H = X.size(2);
    int W = X.size(3);
    int A = D / num_classes_;

    // P->Resize(N * D * H * W);
    // avg_loss->Resize(vector<TIndex>());
    // math::Set<float, CUDAContext>(
    //     avg_loss->size(), 0.f, avg_loss.dptr_, &context_);
    // math::Set<float, CUDAContext>(
    //     P->size(), 0.f, P.dptr_, &context_);
    // math::Set<float, CUDAContext>(
    //     losses_.size(), 0.f, losses_.dptr_, &context_);
    DCHECK_EQ(X.MSize(), 4);

    const DType *Xdata = X.dptr_;

    DType *Pdata = P.dptr_;

    // Spatial Softmax Kernel
    dim3 dimGrid(N * A * H * W);
    dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "SpatialSoftmaxKernel");
    // calculate softmax probabilities: Pdata
    cudaStream_t stream = Stream<gpu>::GetStream(losses_.stream_);
    SpatialSoftmaxKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(
        N, A, H, W, Xdata, Pdata, num_classes_);

    // Compute loss for each x,y location
    const DType *Tdata = T.dptr_;
    const DType *Wdata = wp.dptr_;
    DType *Ldata = losses_.dptr_;

    // dim3 dimGrid(N * A * H * W);
    // dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocalLossKernel");
    // cudaStream_t stream = Stream<gpu>::GetStream(losses_.stream_);
    SoftmaxFocalLossKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(
        N, A, H, W, Pdata, Tdata, Ldata, Wdata, gamma_, alpha_, num_classes_);

    DType *avg_loss_data = avg_loss.dptr_;

    // sum the losses: from losses_ to avg_loss
    mshadow::nansum(avg_loss_data, Ldata);
    // math::Sum<float, CUDAContext>(
    //     losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
    // math::Scale<float, CUDAContext>(
    //     1, scale_, avg_loss_data, avg_loss_data, &context_);
    }


    template<typename DType>
    inline void SoftmaxFocalLossBackwardAcc(const Tensor<gpu, 4, DType> &X, // Logits; data
                                            const Tensor<gpu, 4, DType> &T, // Labels; labels
                                            const Tensor<gpu, 1, DType> &wp, // num of forground ; normalizer
                                            const Tensor<gpu, 4, DType> &P, //softmax probability; prob
                                            const Tensor<gpu, 4, DType> &d_avg_loss, // gradient in
                                            const Tensor<gpu, 4, DType> &dX, // gradient out
                                            const Tensor<gpu, 4, DType> &buff_, // aux buff_ Tensor
                                            const int num_classes_,
                                            const float gamma_,
                                            const float alpha_)
    {
    int N = X.size(0);
    int D = X.size(1);
    int H = X.size(2);
    int W = X.size(3);
    int A = D / num_classes_;

    // buff_.Resize(N * A * H * W);

    // dX->ResizeLike(X);

    //const DType *Xdata = X.dptr_;
    const DType *Tdata = T.dptr_;
    const DType *Pdata = P.dptr_;
    const DType *Wdata = wp.dptr_;

    DType *Bdata = buff_.dptr_;

    // Compute the weight for gradients
    dim3 dimGrid(N * A * H * W);
    dim3 dimBlock(kMaxThreadsPerBlock);
    CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocalLossGradientWeightKernel");
    cudaStream_t stream = Stream<gpu>::GetStream(buff_.stream_);
    SoftmaxFocalLossGradientWeightKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(N, A, H, W, Pdata, Tdata, Bdata,
        Wdata, gamma_, alpha_, num_classes_);


    DType *dXdata = dX.dptr_;
    DType *dALdata = d_avg_loss.dptr_;
    
    // Compute the gradient with the weights
    dim3 dimGrid_(N * D * H * W);
    CheckLaunchParam(dimGrid_, dimBlock, "SoftmaxFocalLossGradientWeightKernel");
    // cudaStream_t stream = Stream<gpu>::GetStream(buff_.stream_);
    SoftmaxFocalLossGradientKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(N, D, H, W, Pdata, Tdata, Bdata, dALdata, dXdata, num_classes_);

    // math::Scale<float, CUDAContext>(
    //     dX->size(), scale_, dX.dptr_, dX.dptr_,
    //     &context_);
    }
} // cuda


    template<typename DType>
    inline void SoftmaxFocalLossForward(const Tensor<gpu, 4, DType> &X, // Logits; data
                                        const Tensor<gpu, 4, DType> &T, // Labels; labels
                                        const Tensor<gpu, 1, DType> &wp, // num of forground ; normalizer
                                        const Tensor<gpu, 1, DType> &avg_loss, // average loss as output; loss
                                        const Tensor<gpu, 4, DType> &P, //softmax probability, going to be re-used in gradient; prob
                                        const Tensor<gpu, 4, DType> &losses_, // aux losses_ Tensor
                                        const int num_classes_,
                                        const float gamma_,
                                        const float alpha_)
    {
        cuda::SoftmaxFocalLossForward(X, T, wp, avg_loss, P, losses_, num_classes_, gamma_, alpha_);
    };

    template<typename DType>
    inline void SoftmaxFocalLossBackwardAcc(const Tensor<gpu, 4, DType> &X, // Logits; data
                                            const Tensor<gpu, 4, DType> &T, // Labels; labels
                                            const Tensor<gpu, 1, DType> &wp, // num of forground ; normalizer
                                            const Tensor<gpu, 4, DType> &P, //softmax probability; prob
                                            const Tensor<gpu, 4, DType> &d_avg_loss, // gradient in
                                            const Tensor<gpu, 4, DType> &dX, // gradient out
                                            const Tensor<gpu, 4, DType> &buff_, // aux buff_ Tensor
                                            const int num_classes_,
                                            const float gamma_,
                                            const float alpha_)
    {
        cuda::SoftmaxFocalLossBackwardAcc(X, T, wp, P, d_avg_loss, dX, buff_, num_classes_, gamma_, alpha_);
    };

} // mshadow



namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(SoftmaxFocalLossParam param, int DType) {
    Operator* op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(DType, DType, {
    op = new SoftmaxFocalLossOp<gpu, DType>(param);
    });
    return op;
}

}  // namespace op
}  // namespace mxnet
