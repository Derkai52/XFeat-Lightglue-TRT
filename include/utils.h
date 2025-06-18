#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <torch/torch.h>

std::vector<cv::Point2f> floatVecToPoints(const std::vector<float>& vec) {
    std::vector<cv::Point2f> pts;
    if (vec.size() % 2 != 0) {
        std::cerr << "关键点坐标数必须为偶数！" << std::endl;
        return pts;
    }
    for (size_t i = 0; i < vec.size(); i += 2) {
        pts.emplace_back(vec[i], vec[i + 1]);
    }
    return pts;
}


static void TensorToVectorKeypoints(const torch::Tensor& keypoints_tensor, std::vector<float>& keypoints_vector) 
{

    // 确保张量是连续的，并且是 float 类型
    torch::Tensor cont_tensor = keypoints_tensor.contiguous().to(torch::kFloat32);
    // 获取数据指针
    const float* data = cont_tensor.data_ptr<float>();
    // 计算元素总数
    size_t num_elements = cont_tensor.numel();
    // 复制数据到 vector
    keypoints_vector = std::vector<float>(data, data + num_elements);

}

// Function to convert a keypoints tensor to std::vector<cv::KeyPoint>
static void TensorToKeypoints(const torch::Tensor& keypoints_tensor, std::vector<cv::KeyPoint>& keypoints_vector) 
{
    // Iterate through the tensor and create KeyPoint objects
    for (int i = 0; i < keypoints_tensor.size(0); ++i) {
        float x = keypoints_tensor[i][0].item<float>();
        float y = keypoints_tensor[i][1].item<float>();

        // Create a KeyPoint object
        cv::KeyPoint kp(x, y, 1.0f);

        // Add the KeyPoint object to the vector
        keypoints_vector.push_back(kp);
    }
}

//Convert Query Tensor and Train Tensor to cv::DMatch type
static void TensorsToDMatch(const torch::Tensor& idx1, const torch::Tensor& idx2, std::vector<cv::DMatch>& matches)
{
    for (int i = 0; i < idx1.size(0); ++i) 
        {
            int queryIdx = idx1[i].item<int>();
            int trainIdx = idx2[i].item<int>();
            float distance = 1.0;
            matches.emplace_back(queryIdx, trainIdx, distance);
        }
}

//Reject outliers from the matching function using RANSAC
static void reject_outliers(const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2, const std::vector<cv::DMatch>& _matches, std::vector<cv::DMatch>& _inliers)
{
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : _matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Use RANSAC to find the fundamental matrix and filter out outliers
    std::vector<uchar> inliersMask(_matches.size());
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(points1, points2, inliersMask, 3, 0.99, cv::FM_RANSAC);

    // Filter matches to retain only inliers
    for (size_t i = 0; i < _matches.size(); ++i) {
        if (inliersMask[i]) {
            _inliers.push_back(_matches[i]);
        }
    }
}

// Function to visualize the keypoints and matching, as well as display some benchmarking results.
static void VisualizeMatching(const cv::Mat &image1, const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &image2, const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &_matches, cv::Mat &output_image, double cost_time = -1) 
{
    cv::Mat img1 = image1;
    cv::Mat img2 = image2;

    if (image1.rows != 480 || image1.cols != 640) 
    {
    std::cout<<"Warning: Image 1 is not of the resolution expected by the visualization function, which is (640x480). Resizing image."<<std::endl;
    cv::resize(img1, img1, cv::Size(640, 480));
    }
    if (image2.rows != 480 || image2.cols != 640) 
    {
    std::cout<<"Warning: Image 2 is not of the resolution expected by the visualization function, which is (640x480). Resizing image."<<std::endl;
    cv::resize(img2, img2, cv::Size(640, 480));
    }
    cv::drawMatches(img1, keypoints1, img2, keypoints2, _matches, output_image, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
    double sc = std::min(img1.rows / 640., 2.0);
    int ht = int(30 * sc);
    std::string title_str = "XFeat TensorRT";
    cv::putText(output_image, title_str, cv::Point(int(8*sc), ht), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, title_str, cv::Point(int(8*sc), ht), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::string feature_points_str = "Keypoints: " + std::to_string(keypoints1.size()) + ":" + std::to_string(keypoints2.size());
    cv::putText(output_image, feature_points_str, cv::Point(int(8*sc), ht*2), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, feature_points_str, cv::Point(int(8*sc), ht*2), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::string match_points_str = "Matches: " + std::to_string(_matches.size());
    cv::putText(output_image, match_points_str, cv::Point(int(8*sc), ht*3), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    cv::putText(output_image, match_points_str, cv::Point(int(8*sc), ht*3), cv::FONT_HERSHEY_DUPLEX,1.0*sc, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    if(cost_time != -1) {
        std::string time_str = "FPS: " + std::to_string(1000 / cost_time);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc,
                    cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::putText(output_image, time_str, cv::Point(int(8 * sc), ht * 4), cv::FONT_HERSHEY_DUPLEX, 1.0 * sc,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

#endif //UTILS_H