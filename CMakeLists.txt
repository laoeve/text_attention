cmake_minimum_required(VERSION 3.14)
project(TransformerNLP CXX)

set(CMAKE_CXX_STANDARD 20)
include(CTest)
enable_testing()

include_directories(model)
add_executable(TransformerNLP main.cpp)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -DDEBUG -Wno-sign-compare -Wno-reorder -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -Wno-sign-compare -Wno-reorder")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
