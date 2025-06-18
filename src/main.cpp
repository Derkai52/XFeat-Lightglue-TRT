#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include "utils.h"
#include "xfeat.h"
#include "lightglue.h"

std::vector<MatchPoint> matches;
std::vector<cv::KeyPoint> k1,k2;
std::vector<cv::Point2f> kpts1, kpts2;
torch::Tensor feats1, keypoints1, heatmap1, idx1;
torch::Tensor feats2, keypoints2, heatmap2, idx2;

// 这里的路径按实际情况修改
std::string image1_path = "/home/tk/xfeat_lightglue_trt/assets/ref.png";
std::string image2_path = "/home/tk/xfeat_lightglue_trt/assets/tgt.png";
std::string config_path = "/home/tk/xfeat_lightglue_trt/config/xfeat_lightglue.yaml";
std::string xfeat_engine_path = "/home/tk/xfeat_lightglue_trt/weights/xfeat_1_800_800.engine";
std::string glue_engine_path = "/home/tk/xfeat_lightglue_trt/weights/lightglue_L6_1_800_800.engine";
// Xfeat 特征提取模块
XFeat xfeat(config_path, xfeat_engine_path);
// LIghtglue 特征匹配模块
Lightglue lightglue(config_path, glue_engine_path);


void feature_extract_match(const cv::Mat& img1, const cv::Mat& img2){
    // auto start = std::chrono::high_resolution_clock::now();
    xfeat.detectAndCompute(img1, keypoints1, feats1, heatmap1); // 提取图1 的特征点和描述子
    xfeat.detectAndCompute(img2, keypoints2, feats2, heatmap2); // 提取图2 的特征点和描述子

    std::vector<float> k1_v, k2_v;
    std::vector<float> f1_v, f2_v;
    TensorToVectorKeypoints(keypoints1, k1_v);
    TensorToVectorKeypoints(keypoints2, k2_v);
    TensorToVectorKeypoints(feats1, f1_v);
    TensorToVectorKeypoints(feats2, f2_v);
    // 匹配
    lightglue.matching(k1_v, k2_v, f1_v, f2_v, matches);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Matching done, time: " << duration << std::endl;

    kpts1 = floatVecToPoints(k1_v);
    kpts2 = floatVecToPoints(k2_v);
}

void draw_match_result(const cv::Mat& img1, const cv::Mat& img2, std::vector<MatchPoint> draw_matches){
    // 估计仿射变换矩阵（只包含旋转、缩放和平移，不包括投影）
    cv::Mat inliers;
    std::vector<cv::Point2f> kpts1_sort, kpts2_sort;
    for (const auto& match : draw_matches) {
        kpts1_sort.push_back(kpts1[match.idx1]);  // idx1是kpts1中点的索引
        kpts2_sort.push_back(kpts2[match.idx2]);  // idx2是kpts2中点的索引
    }
    cv::Mat affine_matrix = cv::findHomography(kpts1_sort, kpts2_sort, cv::RANSAC, 3, inliers);

    // 绘制封闭曲线（绿色线段）
    cv::Mat img1_with_corners = img1.clone();
    cv::Mat img2_with_corners = img2.clone();
    cv::cvtColor(img2_with_corners, img2_with_corners, cv::COLOR_GRAY2BGR); // 把灰度的图像转换为彩色图像才能绘制

    // 将角点转换到第二幅图像空间
    std::vector<cv::Point2f> warped_corners;
    int h = img1_with_corners.rows;
    int w = img1_with_corners.cols;
    std::vector<cv::Point2f> corners_img1 = {
    {0.f,      0.f},
    {w - 1.f,  0.f},
    {w - 1.f,  h - 1.f},
    {0.f,      h - 1.f}
    };
    cv::transform(corners_img1, warped_corners, affine_matrix);

    for (size_t i = 0; i < warped_corners.size(); ++i) {
        cv::Point start_point = warped_corners[(i + warped_corners.size() - 1) % warped_corners.size()];
        cv::Point end_point   = warped_corners[i];
        cv::line(img2_with_corners, start_point, end_point, cv::Scalar(0, 255, 0), 4);
    }

    // 拼接图像
    cv::Mat result_img;
    cv::hconcat(img1_with_corners, img2_with_corners, result_img);
    int offsetX = w;
    for (const auto& match : draw_matches) {
        if (match.idx1 >= kpts1.size() || match.idx2 >= kpts2.size()) continue;
        const auto& pt1 = kpts1[match.idx1];
        const auto& pt2 = kpts2[match.idx2];

        // 生成文本：编号 + 置信度（保留两位小数）
        std::ostringstream oss;
        oss << match.idx1 << " (" << std::fixed << std::setprecision(2) << match.score << ")";
        // 在点的右上方写上索引
        cv::putText(result_img, oss.str(), pt1 + cv::Point2f(5, -5),  // 位置偏移
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,            // 字体和缩放
                    cv::Scalar(0, 0, 255), 1);                // 红色字体


        float s = std::min(std::max(match.score, 0.f), 1.f);
        cv::Scalar color(0, 255 * s, 255 * (1 - s));  // Green → Red
        
        cv::Point2f pt2_shifted = pt2 + cv::Point2f(offsetX, 0);
        cv::circle(result_img, pt1, 3, color, -1);
        cv::circle(result_img, pt2_shifted, 3, color, -1);
        cv::line(result_img, pt1, pt2_shifted, color, 1);
    }
    cv::imshow("Match result", result_img);
    cv::waitKey(1);
}


int main(int argc, char* argv[])
{
    // Synchronize to ensure all operations are complete before starting the timer
    torch::cuda::synchronize();
    cv::Mat img1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(image2_path, cv::IMREAD_GRAYSCALE);
    while (true){
        //  Resize 为模型需要的输入尺寸
        cv::resize(img1, img1, cv::Size(800, 800), 0, 0, cv::INTER_AREA);
        cv::resize(img2, img2, cv::Size(800, 800), 0, 0, cv::INTER_AREA);
        feature_extract_match(img1, img2);
        draw_match_result(img1, img2, matches);
    }
    return 0;
}
