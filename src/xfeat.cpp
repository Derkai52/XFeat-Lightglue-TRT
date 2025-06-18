#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "xfeat.h"
#include <yaml-cpp/yaml.h>

using namespace nvinfer1;

XFeat::XFeat(const std::string config_path, const std::string engine_path):dev(torch::kCUDA)
{
    YAML::Node config = YAML::LoadFile(config_path);

    // XFeat params
    std::string engineFilePath = engine_path;
    inputH = config["image_height"].as<int>();
    inputW = config["image_width"].as<int>();
    top_k = config["max_keypoints"].as<int>();

    // NMS params
    threshold = config["feat_threshold"].as<float>();
    kernel_size = config["kernel_size"].as<int>();

    //Softmax params
    softmaxTemp = config["softmaxTemp"].as<float>();

    // Load and initialize the engine
    loadEngine(engineFilePath);
    context = std::unique_ptr<IExecutionContext, DestroyObjects> (engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }

    // Image height and width after image preprocessing to make it compatible with the TensorRT engine.
    _H = (inputH/32)*32;
    _W = (inputW/32)*32;

    //Size of output of TensorRT engine
    outputH = _H/8;
    outputW = _W/8;

    //Scale correction factor
    rh = static_cast<float>(inputH) / static_cast<float>(_H);
    rw = static_cast<float>(inputW) / static_cast<float>(_W);

    //Get engine bindings
    inputIndex = engine->getBindingIndex("image");
    featsIndex = engine->getBindingIndex("feats");
    keypointsIndex = engine->getBindingIndex("keypoints");
    heatmapIndex = engine->getBindingIndex("heatmap");

    //Sparse interpolator for post-processing outputs
    _nearest = InterpolateSparse2D("nearest");
	bilinear = InterpolateSparse2D("bilinear");

}

void XFeat::detectAndCompute(const cv::Mat& img, torch::Tensor& keypoints, torch::Tensor& descriptors, torch::Tensor& scores)
{

    // Preprocess input image and convert to Tensor on GPU
    torch::Tensor input_Data = preprocessImages(img);

    batchSize = input_Data.size(0);

    // Variables to store output from TensorRT engine
    featsData = torch::empty({batchSize,64,outputH,outputW}, torch::device(dev).dtype(torch::kFloat32));
    keypointsData = torch::empty({batchSize, 65, outputH, outputW}, torch::device(dev).dtype(torch::kFloat32));
    heatmapData = torch::empty({batchSize, 1, outputH, outputW}, torch::device(dev).dtype(torch::kFloat32));

    // Create buffer to store input and outputs of TensorRT engine
    void* buffers[4]; 
    buffers[inputIndex] = input_Data.data_ptr();
    buffers[featsIndex] = featsData.data_ptr();
    buffers[keypointsIndex] = keypointsData.data_ptr();
    buffers[heatmapIndex] = heatmapData.data_ptr();


    // auto start = std::chrono::high_resolution_clock::now();
    
    // Run inference on TensorRT engine
    context->executeV2(buffers);
    
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout<<"Xfeat Inference benchmark done "<< duration.count() << " ms"<< std::endl;


    featsData = torch::nn::functional::normalize(featsData, torch::nn::functional::NormalizeFuncOptions().dim(1));
    keypointsData = get_kpts_heatmap(keypointsData,softmaxTemp);
    auto mkpts = NMS(keypointsData,threshold, kernel_size);

    auto scores_ = (_nearest.forward(keypointsData, mkpts, _H, _W) * bilinear.forward(heatmapData, mkpts, _H, _W)).squeeze(-1);

    // Masking
    auto mask_ = torch::all(mkpts == 0, -1);
    scores_.masked_fill_(mask_,-1);

    // Select top k features
    auto idxs = std::get<1>(torch::sort(-scores_));
    idxs = idxs.slice(-1,0, top_k);
    auto mkpts_x = torch::gather(mkpts.select(-1,0),-1, idxs);
    auto mkpts_y = torch::gather(mkpts.select(-1,1),-1, idxs);
    mkpts = torch::cat(std::vector<torch::Tensor>{mkpts_x.unsqueeze(-1), mkpts_y.unsqueeze(-1)}, -1);
    scores_ = torch::gather(scores_, -1, idxs);

    // Interpolate descriptors
    auto feats = bilinear.forward(featsData, mkpts, _H, _W);

    // L2-Normalize
    feats = torch::nn::functional::normalize(feats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

    // Correct keypoint scale
    auto scale = torch::tensor({rw, rh}, torch::kFloat32).to(dev).view({1, 1, -1});
    mkpts = mkpts * scale;

    // Validity mask
    auto valid = (scores_ > 0);

    // Use the valid mask to filter the keypoints, scores, and descriptors
    auto keypoints_valid = mkpts.index({valid});
    auto scores_valid = scores_.index({valid});
    auto descriptors_valid = feats.index({valid});

    // Synchronize all operations before moving the results back to the CPU
    torch::cuda::synchronize();

    // Transfer the valid points to the CPU
    keypoints = keypoints_valid.cpu();
    scores = scores_valid.cpu();
    descriptors = descriptors_valid.cpu();
}

void XFeat::detectDense(const cv::Mat& img, torch::Tensor& keypoints, torch::Tensor& descriptors)
{
    // Preprocess input image and convert to Tensor on GPU
    torch::Tensor input_Data = preprocessImages(img);

    batchSize = input_Data.size(0);

    // Variables to store output from TensorRT engine
    featsData = torch::empty({batchSize,64,outputH,outputW}, torch::device(dev).dtype(torch::kFloat32));
    keypointsData = torch::empty({batchSize, 65, outputH, outputW}, torch::device(dev).dtype(torch::kFloat32));
    heatmapData = torch::empty({batchSize, 1, outputH, outputW}, torch::device(dev).dtype(torch::kFloat32));

    // Create buffer to store input and outputs of TensorRT engine
    void* buffers[4]; 
    buffers[inputIndex] = input_Data.data_ptr();
    buffers[featsIndex] = featsData.data_ptr();
    buffers[keypointsIndex] = keypointsData.data_ptr();
    buffers[heatmapIndex] = heatmapData.data_ptr();

    // Run inference on TensorRT engine
    context->executeV2(buffers);

    featsData = featsData.permute({0, 2, 3, 1}).reshape({batchSize, -1, 64});
    heatmapData = heatmapData.permute({0, 2, 3, 1}).reshape({batchSize, -1});

    // Create a grid of (x, y) coordinates
    torch::Tensor xy;
    create_xy(outputH, outputW, xy);
    xy = xy.mul(8).expand({batchSize, -1, -1});

    auto [heatmap_topk, top_k_indices] = torch::topk(heatmapData, std::min(int(heatmapData.size(1)), top_k), -1);

    auto feats = torch::gather(featsData, 1, top_k_indices.unsqueeze(-1).expand({-1, -1, 64}));
    auto mkpts = torch::gather(xy, 1, top_k_indices.unsqueeze(-1).expand({-1, -1, 2}));
    mkpts = mkpts * torch::tensor({rw, rh}, dev).view({1, -1});
    
    feats = feats.squeeze(0);
    mkpts = mkpts.squeeze(0);

    // Synchronize operations before transferring results to the CPU
    torch::cuda::synchronize();

    // Return the results to the CPU
    keypoints = mkpts.cpu();
    descriptors = feats.cpu();
}

torch::Tensor XFeat::preprocessImages(const cv::Mat& img)
{
    torch::Tensor img_tensor = MatToTensor(img);

    img_tensor = torch::nn::functional::interpolate(
        img_tensor,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{_H, _W})
            .mode(torch::kBilinear)
            .align_corners(false)
    );


    return img_tensor;
}

std::vector<char> XFeat::readEngineFile(const std::string& engineFilePath)
{
    std::ifstream file(engineFilePath, std::ios::binary | std::ios::ate);
    if(!file.is_open()){
        throw std::runtime_error("Unable to open engine file: " + engineFilePath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0,std::ios::beg);

    std::vector<char> buffer(size);
    if(!file.read(buffer.data(),size)){
        throw std::runtime_error("Unable to read engine file: " + engineFilePath);
    }
    return buffer;
}

void XFeat::loadEngine(const std::string& engineFilePath)
{
    std::vector<char> engineData = readEngineFile(engineFilePath);

    runtime = std::unique_ptr<IRuntime,DestroyObjects>(createInferRuntime(gLogger));

    if(!runtime){
        throw std::runtime_error("Unable to create TensorRT runtime");
    }

    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* rawEngine = runtime->deserializeCudaEngine(engineData.data(),engineData.size());

    if(!rawEngine)
    {
        throw std::runtime_error("Unable to deserialize TensorRT engine");
    }
    engine = std::unique_ptr<ICudaEngine, DestroyObjects>(rawEngine);
}

inline torch::Tensor XFeat::MatToTensor(const cv::Mat& img)
{
    cv::Mat floatMat;
    img.convertTo(floatMat,CV_32F);

    CV_Assert(floatMat.isContinuous());

    int channels = floatMat.channels();
    int height = floatMat.rows;
    int width = floatMat.cols;

    torch::Tensor img_tensor = torch::from_blob(floatMat.data, {1, height, width, channels}, torch::kFloat32);
    img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();
    img_tensor = img_tensor.to(dev);

    return img_tensor;
}

torch::Tensor XFeat::NMS(const torch::Tensor& x, float threshold, int kernel_size)
{
    auto options = torch::TensorOptions().dtype(torch::kLong).device(dev);

    int B = x.size(0);
    int H = x.size(2);
    int W = x.size(3);
    int pad = kernel_size / 2;

    //Perform MaxPool2d
    auto local_max = torch::max_pool2d(x,kernel_size, 1, pad);
    // Compare x with local_max and threshold
    auto pos = (x == local_max) & (x > threshold);
    // Get the positions of the positive elements
    std::vector<torch::Tensor> pos_batched;
    pos_batched.reserve(B);
    for(int i = 0; i < B; i++)
    {
        pos_batched.emplace_back(pos[i].nonzero().slice(/*dim=*/1, /*start=*/1, /*end=*/torch::indexing::None).flip(-1));
    }
    // Find the maximum number of keypoints to pad the tensor
    int pad_val = 0;
    for(const auto& tensor : pos_batched)
    {
        pad_val = std::max(pad_val, static_cast<int>(tensor.size(0)));
    }
    //Pad keypoints and build (B, N, 2) Tensor
    auto pos_tensor = torch::zeros({B, pad_val, 2}, options);
    for(int b = 0; b < B; b++)
    {
        pos_tensor[b].narrow(0, 0, pos_batched[b].size(0)) = pos_batched[b];
    }
    return pos_tensor;
}

torch::Tensor XFeat::get_kpts_heatmap(const torch::Tensor& kpts, float softmax_temp)
{
    //Apply softmax to the input tensor with temperature
    auto scores = torch::softmax(kpts * softmax_temp, 1).narrow(1,0,64);

    //Get dimension
    int B = scores.size(0);
    int H = scores.size(2);
    int W = scores.size(3);
  
    //Perform reshaping and permutation
    auto heatmap = scores.permute({0,2,3,1}).reshape({B, H, W, 8, 8});
    heatmap = heatmap.permute({0,1,3,2,4}).reshape({B, 1, H*8, W*8});

    return heatmap;
}

void XFeat::match(const torch::Tensor& feats1, const torch::Tensor& feats2, torch::Tensor& idx1, torch::Tensor& idx2, double min_cossim) 
{
    auto cossim = torch::matmul(feats1, feats2.t());
    auto cossim_t = torch::matmul(feats2, feats1.t());

    auto match12 = std::get<1>(cossim.max(1));
    auto match21 = std::get<1>(cossim_t.max(1));

    idx1 = torch::arange(match12.size(0), cossim.options().device(match12.device()));
    auto mutual = match21.index({match12}) == idx1;

    if (min_cossim > 0) {
        cossim = std::get<0>(cossim.max(1));
        auto good = cossim > min_cossim;
        idx1 = idx1.index({mutual & good});
        idx2 = match12.index({mutual & good});
    } 
    else 
    {
        idx1 = idx1.index({mutual});
        idx2 = match12.index({mutual});
    }
}

void XFeat::create_xy(int h, int w, torch::Tensor& xy) 
{
    auto y = torch::arange(h, dev).view({-1, 1});
    auto x = torch::arange(w, dev).view({1, -1});
    xy = torch::cat({x.repeat({h, 1}).unsqueeze(-1), y.repeat({1, w}).unsqueeze(-1)}, -1).view({-1, 2});
}