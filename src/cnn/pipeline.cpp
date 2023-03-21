//
// Created by sdy_zjx on 2023/3/1.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "pipeline.h"

namespace {
    void cv_show(const cv::Mat & one_image, const char * info = "") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    bool cv_write(const cv::Mat & source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }
}

using namespace pipeline;

std::map<std::string, pipeline::list_type> pipeline::get_images_for_classification(
        const std::filesystem::path dataset_path,
        const std::vector<std::string> categories,
        const std::pair<float, float> ratios) {
    // 遍历 dataset_path 文件夹下指定的类别
    list_type all_images_list;
    const int categories_num = categories.size();
    for(int i = 0;i < categories_num; ++i) {
        const auto images_dir = dataset_path / categories[i];
        assert(std::filesystem::exists(images_dir) && std::string(images_dir.string() + " 路径不存在!").c_str());
        auto walker = std::filesystem::directory_iterator(images_dir);
        for(const auto& iter : walker)
            all_images_list.emplace_back(iter.path().string(), i);
    }
    // 打乱图像列表
    std::shuffle(all_images_list.begin(), all_images_list.end(), std::default_random_engine(212));
    // 将数据集划分成三部分
    const int total_size = all_images_list.size();
    assert(ratios.first > 0 && ratios.second > 0 && ratios.first + ratios.second < 1);
    const int train_size = int(total_size * ratios.first);
    const int test_size = int(total_size * ratios.second);
    std::map<std::string, list_type> results;
    results.emplace("train", list_type(all_images_list.begin(), all_images_list.begin() + train_size));
    results.emplace("test", list_type(all_images_list.begin() + train_size, all_images_list.begin() + train_size + test_size));
    results.emplace("valid", list_type(all_images_list.begin() + train_size + test_size, all_images_list.end()));
    std::cout << "train  :  " << results["train"].size() << "\n" << "test   :  " << results["test"].size() << "\n" << "valid  :  " << results["valid"].size() << "\n";
    return results;
}


Dataloader::batch_type Dataloader::generate_batch() {
    std::vector<tensor> images;
    std::vector<int> labels;
    images.reserve(this->batch_size);
    labels.reserve(this->batch_size);
    for (int i = 0; i < this->batch_size; ++i) {
        auto sample = this->add_to_buffer(i);
        images.emplace_back(sample.first);
        labels.emplace_back(sample.second);
    }
    return make_pair(std::move(images), std::move(labels));
}
Dataloader::Dataloader(const int bs_, const bool shuffle_, const int _seed, const std::tuple<int, int, int> image_size) : batch_size(bs_), shuffle(shuffle_), seed(_seed),
H(std::get<0>(image_size)), W(std::get<1>(image_size)), C(std::get<2>(image_size)) {
    this->images_num = this->images_list.size();
    this->buffer.reserve(this->batch_size);
    for (int i = 0; i < this->batch_size; ++i) {
        this->buffer.emplace_back(new Tensor3D(C,H,W));
    }
}
int Dataloader::length() const {return this->images_num;}

std::pair<tensor, int> Dataloader::add_to_buffer(const int batch_index) {
    ++this->iter;
    if (this->iter == this->images_num) {
        this->iter = 0;
        if (this->shuffle) {
            std::shuffle(this->images_list.begin(), this->images_list.end(), std::default_random_engine(this->seed));
        }
    }
    const auto& image_path = this->images_list[this->iter].first;
    const int image_label = this->images_list[this->iter].second;
    cv::Mat origin = cv::imread(image_path);
    cv::resize(origin, origin, {W, H});
    this->buffer[batch_index]->read_from_mat(origin.data);
    return std::pair<tensor, int>(this->buffer[batch_index], image_label);
}