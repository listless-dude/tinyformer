cmake_minimum_required(VERSION 3.10)
project(tinyformer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EIGEN3_INCLUDE_DIR "D:/eigen-5.0.1" CACHE PATH "Path to the Eigen root directory")

if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR
        "Eigen headers not found. Set EIGEN3_INCLUDE_DIR to your Eigen root directory "
        "(example: D:/eigen-5.0.1)."
    )
endif()

add_executable(main
    src/main.cpp
)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)