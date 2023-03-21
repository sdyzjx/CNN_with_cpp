//
// Created by sdy_zjx on 2023/3/14.
//
#ifndef CNN_WITH_CPP_ARCHITECTURES_H
#define CNN_WITH_CPP_ARCHITECTURES_H

#include "pipeline.h"
namespace architectures {
    using namespace pipeline;
    extern bool no_grad;
    class WithoutGrad {
    public:
        WithoutGrad() {
            architectures::no_grad = true;
        }
        ~WithoutGrad() noexcept {
            architectures::no_grad = false;
        }
    };

    class Layer {
    public:
        const std::string name;
        std::vector<tensor> output;
        Layer(std::string& _name) : name(std::move(_name)) {}
        virtual std::vector<tensor> forward(const std::vector<tensor>& input) = 0;
        virtual std::vector<tensor> backward(std::vector<tensor>& delta) = 0;
        virtual void save_weights(std::ofstream& writer) const {}
        virtual void load_weights(std::ifstream& reader) {}
        virtual std::vector<tensor> get_output() const { return this->output; }

    };

    class ReLU : public Layer {
    public:
        ReLU(std::string _name) : Layer{_name} {}
        std::vector<tensor> forward(const std::vector<tensor>& input);
        std::vector<tensor> backward(std::vector<tensor>& delta);
    };
}



#endif //CNN_WITH_CPP_ARCHITECTURES_H
