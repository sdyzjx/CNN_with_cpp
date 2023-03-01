#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "./src/cnn/tensor.h"
using namespace std;

int main() {
    auto path = "./src/233.jpg";
    cv::Mat image = cv::imread(path);
    Tensor3D test(3, image.rows, image.cols);
    test.read_from_mat(image.data);
    cv::Mat img_test = test.tensor_mat(3);
    cv::imshow("test", img_test);
    cv::waitKey(0);
    return 0;
}
