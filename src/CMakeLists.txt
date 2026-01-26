cmake_minimum_required(VERSION 3.16)
project(Image_Processing)

# Standard C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inclusion du dossier stb
include_directories(${CMAKE_SOURCE_DIR}/stb)

# Détection d'OpenCV
find_package(OpenCV REQUIRED)

# Activation de OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "OpenMP trouvé, activation du support multithread.")
else()
    message(FATAL_ERROR "OpenMP non trouvé. Installez-le avec 'sudo apt install libomp-dev'.")
endif()

# Exécutable principal
add_executable(video_transform
        Image.cpp
        Image.h
        ImageProcessing.cpp
        ImageProcessing.h
        main.cpp
        schrift.h
        TransformationsConfig.h
        VideoProcessing.cpp
        VideoProcessing.h
)

# Lien avec OpenCV et OpenMP
target_link_libraries(video_transform PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# Détection de support AVX2 et optimisations pour ton processeur (Intel i5-7260U)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
CHECK_CXX_COMPILER_FLAG("-fopenmp" COMPILER_SUPPORTS_OPENMP)

if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 supporté, optimisation pour ton processeur.")
    target_compile_options(video_transform PRIVATE
            -O3
            -march=native
            -mavx2
            -mfma
    )
else()
    message(WARNING "AVX2 non supporté, optimisation générique appliquée.")
    target_compile_options(video_transform PRIVATE -O3)
endif()

if(COMPILER_SUPPORTS_OPENMP)
    message(STATUS "OpenMP supporté, activation des flags de compilation.")
    target_compile_options(video_transform PRIVATE -fopenmp)
else()
    message(FATAL_ERROR "Le compilateur ne supporte pas OpenMP. Vérifie ton installation de GCC.")
endif()
