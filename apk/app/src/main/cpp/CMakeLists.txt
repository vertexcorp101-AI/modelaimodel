cmake_minimum_required(VERSION 3.22.1)
project(tinyvertex_llama)

# llama.cpp source is cloned at build time by the Colab script
# into ${LLAMA_SRC_DIR} (default: ./llama.cpp)
set(LLAMA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

if(NOT EXISTS ${LLAMA_SRC_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_SRC_DIR}. Clone it before building.")
endif()

# Build llama.cpp as a static library
set(LLAMA_BUILD_SHARED_LIBS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_SERVER OFF)
set(GGML_OPENMP OFF)
set(GGML_NATIVE OFF)
add_subdirectory(${LLAMA_SRC_DIR} llama_build)

# JNI wrapper
add_library(tinyvertex_jni SHARED
    tinyvertex_jni.cpp
)

target_include_directories(tinyvertex_jni PRIVATE
    ${LLAMA_SRC_DIR}/include
    ${LLAMA_SRC_DIR}/ggml/include
)

target_link_libraries(tinyvertex_jni
    llama
    android
    log
)
