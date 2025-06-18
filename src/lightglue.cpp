#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "lightglue.h"
#include <yaml-cpp/yaml.h>

using namespace nvinfer1;

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "CUDA failure: " << ret << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)


using namespace nvinfer1;

void printBindingsInfo(nvinfer1::ICudaEngine* engine) {
    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getBindingName(i);
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        bool isInput = engine->bindingIsInput(i);

        std::cout << (isInput ? "[Input] " : "[Output] ") << name << ": (";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
}

Lightglue::Lightglue(const std::string config_path, const std::string engine_path):dev(torch::kCUDA)
{
    YAML::Node config = YAML::LoadFile(config_path);

    // Lightglue params
    std::string engineFilePath = engine_path;
    maxMatches = config["max_matches"].as<int>();
    threShold = config["match_threshold"].as<double>();

    // Load and initialize the engine
    loadEngine(engineFilePath);
    context = std::unique_ptr<IExecutionContext, GlueDestroyObjects> (engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }

    //Get engine bindings
    image0_size_Index = engine->getBindingIndex("image0_size");
    image1_size_Index = engine->getBindingIndex("image1_size");
    keypoints_0_Index = engine->getBindingIndex("mkpts0");
    keypoints_1_Index = engine->getBindingIndex("mkpts1");
    descriptors_0_Index = engine->getBindingIndex("feats0");
    descriptors_1_Index = engine->getBindingIndex("feats1");
    matches_Index = engine->getBindingIndex("matches");
    scores_Index = engine->getBindingIndex("scores");


    int kpt_num = 512;

    // 输入维度在这里设置
    nvinfer1::Dims dim1;
    dim1.nbDims = 1;      // 1个维度
    dim1.d[0] = 2;        // 值为2

    nvinfer1::Dims dim2;
    dim2.nbDims = 1;      // 1个维度
    dim2.d[0] = 2;        // 值为2

    context->setBindingDimensions(image0_size_Index, dim1);
    context->setBindingDimensions(image1_size_Index, dim2);
    context->setBindingDimensions(keypoints_0_Index, nvinfer1::Dims3{1, kpt_num, 2});
    context->setBindingDimensions(keypoints_1_Index, nvinfer1::Dims3{1, kpt_num, 2});
    context->setBindingDimensions(descriptors_0_Index, nvinfer1::Dims3{1, kpt_num, 64});
    context->setBindingDimensions(descriptors_1_Index, nvinfer1::Dims3{1, kpt_num, 64});
    assert(context->allInputDimensionsSpecified());

}

void Lightglue::matching(std::vector<float> keypoints1, std::vector<float> keypoints2, std::vector<float> feats1, std::vector<float> feats2, std::vector<MatchPoint>& matches)
{
    size_t img0_size = 2 * sizeof(float);
    size_t img1_size = 2 * sizeof(float);
    size_t kpt_num = 512;
    size_t kpts_size = 1 * kpt_num * 2 * sizeof(int);
    size_t desc_size = 1 * kpt_num * 64 * sizeof(float);
    size_t max_matches = maxMatches;
    size_t match_output_size = max_matches * 2 * sizeof(int);
    size_t score_output_size = max_matches * sizeof(float);

    float *d_imgsize0, *d_imgsize1, *d_kpts0, *d_kpts1, *d_desc0, *d_desc1, *d_scores;
    int *d_matches;
    CHECK(cudaMalloc((void**)&d_imgsize0, img0_size));
    CHECK(cudaMalloc((void**)&d_imgsize1, img1_size));
    CHECK(cudaMalloc((void**)&d_kpts0, kpts_size));
    CHECK(cudaMalloc((void**)&d_kpts1, kpts_size));
    CHECK(cudaMalloc((void**)&d_desc0, desc_size));
    CHECK(cudaMalloc((void**)&d_desc1, desc_size));
    CHECK(cudaMalloc((void**)&d_matches, match_output_size));
    CHECK(cudaMalloc((void**)&d_scores, score_output_size));

    CHECK(cudaMemcpy(d_imgsize0, std::vector<float>{800.0f, 800.0f}.data(), img0_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_imgsize1, std::vector<float>{800.0f, 800.0f}.data(), img1_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kpts0, keypoints1.data(), kpts_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kpts1, keypoints2.data(), kpts_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_desc0, feats1.data(), desc_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_desc1, feats2.data(), desc_size, cudaMemcpyHostToDevice));
    
    void* bindings[8];
    bindings[image0_size_Index] = d_imgsize0;
    bindings[image1_size_Index] = d_imgsize1;
    bindings[keypoints_0_Index] = d_kpts0;
    bindings[keypoints_1_Index] = d_kpts1;
    bindings[descriptors_0_Index] = d_desc0;
    bindings[descriptors_1_Index] = d_desc1;
    bindings[matches_Index] = d_matches;
    bindings[scores_Index] = d_scores;

    // auto start = std::chrono::high_resolution_clock::now();

    // Run inference on TensorRT engine
    context->enqueueV2(bindings, 0, nullptr);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout<<"LightGlue Inference benchmark done "<< duration.count() << " ms"<< std::endl;

    // Dims match_dims = context->getBindingDimensions(matches_Index);
    nvinfer1::Dims match_dims = context->getTensorShape("matches");
    Dims score_dims = context->getBindingDimensions(scores_Index);
    int num_matches = match_dims.d[0];

    std::vector<int> h_matches(num_matches * 2);
    std::vector<float> h_scores(num_matches);

    CHECK(cudaMemcpy(h_matches.data(), d_matches, num_matches * 2 * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_scores.data(), d_scores, num_matches * sizeof(float), cudaMemcpyDeviceToHost));

    // 过滤低分匹配对，并保存
    for (int i = 0; i < num_matches; ++i) {
        // std::cout << "Match[" << i << "] = (" << h_matches[i * 2]
        //         << ", " << h_matches[i * 2 + 1] << "), Score = " << h_scores[i] << std::endl;
        if (h_scores[i] >= threShold){
            matches.push_back({h_matches[i * 2], h_matches[i * 2 + 1], h_scores[i]});
        }
    }

    cudaFree(d_imgsize0);
    cudaFree(d_imgsize1);
    cudaFree(d_kpts0);
    cudaFree(d_kpts1);
    cudaFree(d_desc0);
    cudaFree(d_desc1);
    cudaFree(d_matches);
    cudaFree(d_scores);
}


std::vector<char> Lightglue::readEngineFile(const std::string& engineFilePath)
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

void Lightglue::loadEngine(const std::string& engineFilePath)
{
    std::vector<char> engineData = readEngineFile(engineFilePath);

    runtime = std::unique_ptr<IRuntime,GlueDestroyObjects>(createInferRuntime(gLogger));

    if(!runtime){
        throw std::runtime_error("Unable to create TensorRT runtime");
    }

    bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    ICudaEngine* rawEngine = runtime->deserializeCudaEngine(engineData.data(),engineData.size());

    // printBindingsInfo(rawEngine);

    if(!rawEngine)
    {
        throw std::runtime_error("Unable to deserialize TensorRT engine");
    }
    engine = std::unique_ptr<ICudaEngine, GlueDestroyObjects>(rawEngine);
}

// inline torch::Tensor Lightglue::MatToTensor(const cv::Mat& img)
// {
//     cv::Mat floatMat;
//     img.convertTo(floatMat,CV_32F);

//     CV_Assert(floatMat.isContinuous());

//     int channels = floatMat.channels();
//     int height = floatMat.rows;
//     int width = floatMat.cols;

//     torch::Tensor img_tensor = torch::from_blob(floatMat.data, {1, height, width, channels}, torch::kFloat32);
//     img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();
//     img_tensor = img_tensor.to(dev);

//     return img_tensor;
// }

// torch::Tensor Lightglue::get_kpts_heatmap(const torch::Tensor& kpts, float softmax_temp)
// {
//     //Apply softmax to the input tensor with temperature
//     auto scores = torch::softmax(kpts * softmax_temp, 1).narrow(1,0,64);

//     //Get dimension
//     int B = scores.size(0);
//     int H = scores.size(2);
//     int W = scores.size(3);
  
//     //Perform reshaping and permutation
//     auto heatmap = scores.permute({0,2,3,1}).reshape({B, H, W, 8, 8});
//     heatmap = heatmap.permute({0,1,3,2,4}).reshape({B, 1, H*8, W*8});

//     return heatmap;
// }

// void Lightglue::match(const torch::Tensor& feats1, const torch::Tensor& feats2, torch::Tensor& idx1, torch::Tensor& idx2, double min_cossim) 
// {
//     auto cossim = torch::matmul(feats1, feats2.t());
//     auto cossim_t = torch::matmul(feats2, feats1.t());

//     auto match12 = std::get<1>(cossim.max(1));
//     auto match21 = std::get<1>(cossim_t.max(1));

//     idx1 = torch::arange(match12.size(0), cossim.options().device(match12.device()));
//     auto mutual = match21.index({match12}) == idx1;

//     if (min_cossim > 0) {
//         cossim = std::get<0>(cossim.max(1));
//         auto good = cossim > min_cossim;
//         idx1 = idx1.index({mutual & good});
//         idx2 = match12.index({mutual & good});
//     } 
//     else 
//     {
//         idx1 = idx1.index({mutual});
//         idx2 = match12.index({mutual});
//     }
// }

