//
// Created by sdy_zjx on 2023/3/1.
//

#ifndef CNN_WITH_CPP_PIPELINE_H
#define CNN_WITH_CPP_PIPELINE_H

#include <map>
#include <random>
#include <filesystem>
#include "tensor.h"

namespace pipeline {
    //构造Dataloader,用于分batch训练，打乱等功能
    using list_type = std::vector<std::pair<std::string, int>>; //string:dataset_path, int:图像分类
    std::map<std::string, list_type> get_images_for_classification(
            const std::filesystem::path dataset_path,
            const std::vector<std::string> categotrie = {},
            const std::pair<float, float> ratios = {0.8, 0.2} //用于分割测试集与训练集的比例
            );
    class Dataloader {
        using batch_type = std::pair<std::vector<tensor>, std::vector<int>>; // batch 是一个 pair
    private:
        list_type images_list;
        int images_num; // 子数据集一共有多少张图像和对应的标签
        const int batch_size;
        const bool shuffle;
        const int seed;
        int iter = -1;
        std::vector<tensor> buffer;
        const int H, W, C;
    public:
        Dataloader(const int bs_ = 1, const bool shuffle_ = false, const int seed = 11445, const std::tuple<int, int, int> image_size = {224,224,3});
        batch_type generate_batch();
        int length() const;
    private:
        std::pair<tensor, int> add_to_buffer(const int batch_index);
    };
}


#endif //CNN_WITH_CPP_PIPELINE_H
