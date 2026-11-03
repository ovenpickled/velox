cmake_minimum_required(VERSION 3.16)
project(llm-engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

include_directories(include external)

set(CORE_SOURCES
    src/model.cpp
    src/ops.cpp
    src/safetensors.cpp
    src/sampler.cpp
    src/tensor.cpp
    src/tokenizer.cpp
)

add_executable(inference src/main.cpp ${CORE_SOURCES})
target_link_libraries(inference PRIVATE OpenMP::OpenMP_CXX pthread)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/eval_perplexity.cpp")
    add_executable(eval_perplexity tools/eval_perplexity.cpp ${CORE_SOURCES})
    target_link_libraries(eval_perplexity PRIVATE OpenMP::OpenMP_CXX pthread)
endif()
