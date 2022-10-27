cmake_minimum_required(VERSION 3.14)
project(Attention_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
include(CTest)
enable_testing()

include_directories(model)
add_executable(Attention_cpp main.cpp)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)





