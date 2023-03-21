//
// Created by sdy_zjx on 2023/2/28.
//
#include "tensor.h"
using namespace std;

void Tensor3D::read_from_mat(const uchar * const img_ptr) {
    const int data_length = H * W;
    float scale = 1.f / 255; //归一化
    for (int i = 0; i < data_length; ++i) {
        const int p = 3 * i; //三个通道，乘以3
        this->data[i] = img_ptr[p] * scale; //直接把uchar类型数据乘以1/255(float)实现归一化与数据类型转换
        this->data[data_length + i] = img_ptr[p+1] * scale;
        this->data[2*data_length + i] = img_ptr[p+2] * scale;
    }
}
//图像在opencv Mat中以连续的内存(数组)储存。以一个3通道彩色图像的Mat数据为例，数据以[B1,G1,R1,B2,G2,R2...]的格式储存。

cv::Mat Tensor3D::tensor_mat(const int CH) const{
    cv::Mat origin;
    if (CH == 3) {
        origin = cv::Mat(H, W, CV_8UC3);
        const int length = H * W;
        for (int i = 0; i < length; ++i) {
            const int p = 3 * i;
            origin.data[p] = cv::saturate_cast<uchar>(255 * data[i]);
            origin.data[p+1] = cv::saturate_cast<uchar>(255 * data[i + length]);
            origin.data[p+2] = cv::saturate_cast<uchar>(255 * data[i + 2 * length]);
        }
    }
    else if (CH == 1) {
        origin = cv::Mat(H, W, CV_8UC1);
        const int length = H * W;
        for (int i = 0; i < length; ++i) {
            origin.data[i] = cv::saturate_cast<uchar>(255 * data[i]);
        }
    }
    return origin;
}
//将tensor中的图像恢复成mat格式