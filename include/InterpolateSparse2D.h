#ifndef INTER_SPARSE_2D_H
#define INTER_SPARSE_2D_H

#include <torch/torch.h>
#include <iostream>

namespace F = torch::nn::functional;

class InterpolateSparse2D : public torch::nn::Module 
{
    public:
    InterpolateSparse2D(const std::string& mode = "bilinear", bool align_corners = false)
    {
        if(mode == "nearest")
        {
            options_ = F::GridSampleFuncOptions().mode(torch::kNearest).padding_mode(torch::kZeros).align_corners(align_corners);
        }

        else
        {
            options_ = F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(align_corners);
        }
    }

    torch::Tensor normgrid(const torch::Tensor& x, int64_t H, int64_t W)
    {
        //Normalize coordinates to [-1,1]
        auto scale = torch::tensor({W-1, H-1}, x.device()).to(x.dtype());
        return 2.0*(x / scale) - 1.0;
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& pos, int64_t H, int64_t W)
    {
        //Normalize positions
        auto grid = normgrid(pos, H, W).unsqueeze(-2).to(x.dtype());

        //Perform grid sampling
        auto sampled = F::grid_sample(x, grid, options_);

        //Permute and squeeze the result to match the expected shape
        return sampled.permute({0, 2, 3, 1}).squeeze(-2);
    }

    private:
    F::GridSampleFuncOptions options_;
};

#endif //INTER_SPARSE_2D_H