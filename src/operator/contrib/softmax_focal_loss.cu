// Modified from Official Caffe2 implementation
// Author: ddlee, me@ddlee.cn

#include "./softmax_focal_loss-inl.h"
#include <mshadow/tensor.h>
#include "../mshadow_op.h"

namespace mshadow {
namespace cuda {

    template<typename DType>
    __global__ void SoftmaxKernel(const int N, const int A,
        const int num_classes, const DType *Xdata, DType *Pdata) {
        CUDA_KERNEL_LOOP(index, N * A) {
        int D = num_classes * A;
        int a = index % A;
        int i = index / A;

        // Subtract max on each cell for numerical reasons
        float max_val = -FLT_MAX;
        for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
            int idx = i * D +  c;
            max_val = max(max_val, Xdata[idx]);
        }
        // Exponentiate
        float expsum = 0.0f;
        for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
            int idx = i * D + c;
            float expx = exp(Xdata[idx] - max_val);
            Pdata[idx] = expx;
            expsum += expx;
        }
        // Normalize
        for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
            int idx = i * D + c;
            Pdata[idx] /= expsum;
        }
        }
    }

    template<typename DType>
    __global__ void SoftmaxFocalLossKernel(
        const int N, const int A, const int num_classes,
        const DType *Pdata, const DType *targets, DType *losses,
        const int valid_cnt, const float gamma, const float alpha) {
        CUDA_KERNEL_LOOP(i, N * A) {
        int D = A * num_classes;
        int a = i % A;
        int n = i / A;
        const int label = static_cast<int>(targets[i]);

        float Np = max(valid_cnt, 1);
        float z = (label == 0) * (1 - alpha) / Np +
                    (label >= 1) * alpha / Np;

        losses[i] = 0.0;
        if (label >= 0) {
            int offset = a * num_classes;
            int idx = n * D + offset + label;
            losses[i] =
                -(pow(1.0 - Pdata[idx], gamma) *
                log(max(Pdata[idx], FLT_MIN))) * z;
        }
        }
    }

    template<typename DType>
    __global__ void SoftmaxFocalLossGradientWeightKernel(
        const int N, const int A, const int num_classes,
        const DType *Pdata, const DType *targets, DType *buff,
        const int valid_cnt, const float gamma, const float alpha) {
        CUDA_KERNEL_LOOP(i, N * A) {
        int D = A * num_classes;
        int a = i % A;
        int n = i / A;
        const int label = static_cast<int>(targets[i]);
        float Np = max(valid_cnt, 1);
        float z =  (label == 0) * (1 - alpha) / Np +
                    (label >= 1) * alpha / Np;

        buff[i] = 0.0;
        if (label >= 0) {
            int offset = a * num_classes;
            int idx = n * D + offset + label;
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
        const int N, const int D, const int num_classes,
        const DType *Pdata, const DType *targets, const DType *buff,
        const DType *d_loss_data, DType *dX) {
        CUDA_KERNEL_LOOP(i, N * D) {
        int A = D / num_classes;
        int d = i % D;
        int a = d / num_classes;
        int c = d % num_classes;
        int n = i / D;
        float d_loss = *d_loss_data;

        int ind = n * A + a;
        const int label = static_cast<int>(targets[ind]);

        float c1 = (label >= 0) * 1.0;
        float c2 = (label == c) * 1.0;
        dX[i] = 0.0;
        dX[i] = c1 * d_loss * buff[ind] * (c2 - Pdata[i]);
        }
    }

    template<typename DType>
    inline void SoftmaxFocalLossForward(const Tensor<gpu, 3, DType> &X, // Logits; data
                                        const Tensor<gpu, 2, DType> &T, // Labels; labels
                                        const Tensor<gpu, 2, DType> &loss,
                                        const Tensor<gpu, 3, DType> &P, //softmax probability, going to be re-used in gradient; prob
                                        const int valid_cnt_,
                                        const float gamma_,
                                        const float alpha_) {
        int N = X.size(0); // batch
        int num_classes = X.size(1); // num of class
        int A = X.size(2); // third dimension, num of anchor in detection case

        const DType *Xdata = X.dptr_;

        DType *Pdata = P.dptr_;

        // Softmax Kernel
        dim3 dimGrid(N * A);
        dim3 dimBlock(kMaxThreadsPerBlock);
        CheckLaunchParam(dimGrid, dimBlock, "SoftmaxKernel");
        // calculate softmax probabilities: Pdata
        cudaStream_t stream = Stream<gpu>::GetStream(loss.stream_);
        SoftmaxKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(
            N, A, num_classes, Xdata, Pdata);

        // Compute loss for each x,y location
        const DType *Tdata = T.dptr_;
        DType *Ldata = loss.dptr_;


        CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocalLossKernel");
        // cudaStream_t stream = Stream<gpu>::GetStream(losses_.stream_);
        SoftmaxFocalLossKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(
            N, A, num_classes, Pdata, Tdata, Ldata, valid_cnt_, gamma_, alpha_);

    }


    template<typename DType>
    inline void SoftmaxFocalLossBackwardAcc(const Tensor<gpu, 3, DType> &X, // Logits; data
                                            const Tensor<gpu, 2, DType> &T, // Labels; labels
                                            const Tensor<gpu, 3, DType> &P, //softmax probability; prob
                                            const Tensor<gpu, 3, DType> &dX, // gradient in
                                            const Tensor<gpu, 2, DType> &dloss, // gradient out
                                            const Tensor<gpu, 2, DType> &buff_, // aux buff_ Tensor
                                            const int valid_cnt_,
                                            const float gamma_,
                                            const float alpha_) {
        int N = X.size(0);
        int num_classes = X.size(1);
        int A = X.size(2);

        const DType *Tdata = T.dptr_;
        const DType *Pdata = P.dptr_;

        DType *Bdata = buff_.dptr_;

        // Compute the weight for gradients
        dim3 dimGrid(N * A);
        dim3 dimBlock(kMaxThreadsPerBlock);
        CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocalLossGradientWeightKernel");
        cudaStream_t stream = Stream<gpu>::GetStream(dX.stream_);
        SoftmaxFocalLossGradientWeightKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(N, A, num_classes, Pdata, Tdata, Bdata,
            valid_cnt_, gamma_, alpha_);

        int D = A * num_classes;
        DType *dXdata = dX.dptr_;
        DType *dLdata = dloss.dptr_;
        
        // Compute the gradient with the weights
        dim3 dimGrid_(N * D);
        CheckLaunchParam(dimGrid_, dimBlock, "SoftmaxFocalLossGradientWeightKernel");
        SoftmaxFocalLossGradientKernel<DType><<<dimGrid_, dimBlock, 0, stream>>>(N, D, num_classes, Pdata, Tdata, Bdata, dLdata, dXdata);

    }
} // cuda


    template<typename DType>
    inline void SoftmaxFocalLossForward(const Tensor<gpu, 3, DType> &X, // Logits; data
                                        const Tensor<gpu, 2, DType> &T, // Labels; labels
                                        const Tensor<gpu, 2, DType> &loss, // aux losses_ Tensor
                                        const Tensor<gpu, 3, DType> &P, //softmax probability, going to be re-used in gradient; prob
                                        const int valid_cnt_,
                                        const float gamma_,
                                        const float alpha_)
    {
        cuda::SoftmaxFocalLossForward(X, T, loss, P, valid_cnt_, gamma_, alpha_);
    };

    template<typename DType>
    inline void SoftmaxFocalLossBackwardAcc(const Tensor<gpu, 3, DType> &X, // Logits; data
                                            const Tensor<gpu, 2, DType> &T, // Labels; labels
                                            const Tensor<gpu, 3, DType> &P, //softmax probability; prob
                                            const Tensor<gpu, 3, DType> &dX, // gradient out
                                            const Tensor<gpu, 2, DType> &dloss, // gradient in
                                            const Tensor<gpu, 2, DType> &buff_, // aux buff_ Tensor
                                            const int valid_cnt_,
                                            const float gamma_,
                                            const float alpha_)
    {
        cuda::SoftmaxFocalLossBackwardAcc(X, T, P, dX, dloss, buff_, valid_gt_, gamma_, alpha_);
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
