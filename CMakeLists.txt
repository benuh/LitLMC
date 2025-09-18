cmake_minimum_required(VERSION 3.16)
project(LitLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(include)

set(SOURCES
    src/main.cpp
    src/text_processor.cpp
    src/neural_network.cpp
    src/tokenizer.cpp
    src/trainer.cpp
    src/inference.cpp
)

add_executable(LitLM ${SOURCES})

target_link_libraries(LitLM Eigen3::Eigen)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(LitLM PRIVATE -g -O0)
else()
    target_compile_options(LitLM PRIVATE -O3)
endif()