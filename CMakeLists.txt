cmake_minimum_required(VERSION 3.24)
project(CNN_with_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(CNN_with_cpp main.cpp src/cnn/tensor.h src/cnn/tensor.cpp)
set(OpenCV_DIR D:/Program/opencv-build/MinGW-build/install)
find_package(OpenCV REQUIRED)
target_link_libraries(CNN_with_cpp ${OpenCV_LIBS})