cmake_minimum_required(VERSION 4.1)
project(Image_Processing)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/stb)


add_executable(Image_Processing
        Image.cpp
        Image.h
        ImageProcessing.cpp
        ImageProcessing.h
        main.cpp
        schrift.h)
