cmake_minimum_required(VERSION 3.16)
project(llm-engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

include_directories(include external)

file(GLOB SOURCES "src/*.cpp")

add_executable(inference ${SOURCES})
target_link_libraries(inference PRIVATE OpenMP::OpenMP_CXX pthread)
