//
// Created by sdy_zjx on 2023/2/27.
//
//The target of this module is to define a Tensor with cpp.
#ifndef CNN_WITH_CPP_TENSOR_H
#define CNN_WITH_CPP_TENSOR_H
#include <opencv2/core.hpp>
using namespace std;
class Tensor3D {
public:
    const int C, H, W; //C:Channels, H:Height, W:Weight
    float * data;
    const string name;
    Tensor3D(const int _C, const int _H, const int _W, const string _name = "pipeline"): C(_C), H(_H), W(_W), data(new float[_C * _H * _W]), name(move(_name)) {}
    Tensor3D(const tuple<int, int, int>& shape, const string _name="pipeline") : C(get<0>(shape)), H(get<1>(shape)), W(get<2>(shape)), data(new float[get<0>(shape) * get<1>(shape) * get<2>(shape)]) {}
    Tensor3D(const int length, const string _name="pipeline") : C(length), H(1), W(1), data(new float[length]), name(move(_name)) {}

};
#endif //CNN_WITH_CPP_TENSOR_H
