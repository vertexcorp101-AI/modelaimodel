cmake_minimum_required(VERSION 3.22.1)
project(tinyvertex_llama C CXX)

# llama.cpp source is cloned into this directory at build time (see BUILD_COLAB.md).
set(LLAMA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

if(NOT EXISTS ${LLAMA_SRC_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp not found at ${LLAMA_SRC_DIR}. Run the clone step from apk/BUILD_COLAB.md first.")
endif()

# ---- llama.cpp build options (must be set BEFORE add_subdirectory) ----
set(BUILD_SHARED_LIBS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON      OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL              OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP             OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE             OFF CACHE BOOL "" FORCE)
set(GGML_BLAS               OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_SRC_DIR} llama_build)

# ---- JNI wrapper ----
add_library(tinyvertex_jni SHARED
    tinyvertex_jni.cpp
)

target_include_directories(tinyvertex_jni PRIVATE
    ${LLAMA_SRC_DIR}/include
    ${LLAMA_SRC_DIR}/ggml/include
)

target_compile_options(tinyvertex_jni PRIVATE -Os -fvisibility=hidden -fvisibility-inlines-hidden)

target_link_libraries(tinyvertex_jni
    llama
    android
    log
)
