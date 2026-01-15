cmake_minimum_required(VERSION 3.16)
project(Image_Processing)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/stb)

add_executable(Image_Processing
        Image.cpp
        Image.h
        ImageProcessing.cpp
        ImageProcessing.h
        main.cpp
        schrift.h
        TransformationsConfig.h
)

target_compile_options(Image_Processing PRIVATE
        -O3
        -march=native
)
