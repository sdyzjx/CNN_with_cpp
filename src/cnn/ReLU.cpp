//
// Created by sdy_zjx on 2023/3/21.
//
#include "architectures.h"
using namespace architectures;
std::vector<tensor> ReLU::forward(const std::vector<tensor>& input) {
    const int batch_size = input.size();
    if (output.empty()) {
        this->output.reserve(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            this->output.emplace_back(new Tensor3D(input[0]->C, input[0]->H, input[0]->W));
        }
    }
}