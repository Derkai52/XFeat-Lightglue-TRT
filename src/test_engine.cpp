#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "CUDA failure: " << ret << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

using namespace nvinfer1;


// 简单日志器
class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

// 读取序列化的 engine 文件
ICudaEngine* loadEngine(const std::string& engine_path, IRuntime* runtime) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return nullptr;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    return runtime->deserializeCudaEngine(engine_data.data(), size, nullptr);
}

void doInference(ICudaEngine* engine, IExecutionContext* context) {
    int kpts0Idx = engine->getBindingIndex("kpts0");
    int kpts1Idx = engine->getBindingIndex("kpts1");
    int desc0Idx = engine->getBindingIndex("desc0");
    int desc1Idx = engine->getBindingIndex("desc1");
    int matchesIdx = engine->getBindingIndex("matches");
    int scoresIdx = engine->getBindingIndex("scores");

    int kpt_num = 512;

    context->setBindingDimensions(kpts0Idx, Dims3{1, kpt_num, 2});
    context->setBindingDimensions(kpts1Idx, Dims3{1, kpt_num, 2});
    context->setBindingDimensions(desc0Idx, Dims3{1, kpt_num, 64});
    context->setBindingDimensions(desc1Idx, Dims3{1, kpt_num, 64});
    assert(context->allInputDimensionsSpecified());

    size_t kpts_size = 1 * kpt_num * 2 * sizeof(float);
    size_t desc_size = 1 * kpt_num * 64 * sizeof(float);
    size_t max_matches = 512;
    size_t match_output_size = max_matches * 2 * sizeof(float);
    size_t score_output_size = max_matches * sizeof(float);

    float *d_kpts0, *d_kpts1, *d_desc0, *d_desc1, *d_matches, *d_scores;
    CHECK(cudaMalloc((void**)&d_kpts0, kpts_size));
    CHECK(cudaMalloc((void**)&d_kpts1, kpts_size));
    CHECK(cudaMalloc((void**)&d_desc0, desc_size));
    CHECK(cudaMalloc((void**)&d_desc1, desc_size));
    CHECK(cudaMalloc((void**)&d_matches, match_output_size));
    CHECK(cudaMalloc((void**)&d_scores, score_output_size));

    std::vector<float> h_kpts0(kpt_num * 2, 0.1f);
    std::vector<float> h_kpts1(kpt_num * 2, 0.2f);
    std::vector<float> h_desc0(kpt_num * 64, 0.3f);
    std::vector<float> h_desc1(kpt_num * 64, 0.4f);

    CHECK(cudaMemcpy(d_kpts0, h_kpts0.data(), kpts_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kpts1, h_kpts1.data(), kpts_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_desc0, h_desc0.data(), desc_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_desc1, h_desc1.data(), desc_size, cudaMemcpyHostToDevice));

    void* bindings[6];
    bindings[kpts0Idx] = d_kpts0;
    bindings[kpts1Idx] = d_kpts1;
    bindings[desc0Idx] = d_desc0;
    bindings[desc1Idx] = d_desc1;
    bindings[matchesIdx] = d_matches;
    bindings[scoresIdx] = d_scores;

    context->enqueueV2(bindings, 0, nullptr);

    Dims match_dims = context->getBindingDimensions(matchesIdx);
    Dims score_dims = context->getBindingDimensions(scoresIdx);
    int num_matches = match_dims.d[0];

    std::vector<float> h_matches(num_matches * 2);
    std::vector<float> h_scores(num_matches);

    CHECK(cudaMemcpy(h_matches.data(), d_matches, num_matches * 2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_scores.data(), d_scores, num_matches * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_matches; ++i) {
        std::cout << "Match[" << i << "] = (" << h_matches[i * 2]
                  << ", " << h_matches[i * 2 + 1] << "), Score = " << h_scores[i] << std::endl;
    }

    cudaFree(d_kpts0);
    cudaFree(d_kpts1);
    cudaFree(d_desc0);
    cudaFree(d_desc1);
    cudaFree(d_matches);
    cudaFree(d_scores);
}

int main(int argc, char** argv) {

    std::string engine_path = "/home/tk/GNSS-Denial-UAV-Location/src/match_location/scripts/weights/lighterglue_L3.engine";
    CHECK(cudaSetDevice(0));

    // Create runtime
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = loadEngine(engine_path, runtime);
    if (!engine) return -1;

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) return -1;

    doInference(engine, context);

    // Cleanup
    context->destroy();
    engine->destroy();
    runtime->destroy();


    return 0;
}