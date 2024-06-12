cmake_minimum_required(VERSION 3.28)

set(TORCH_CUDA_ARCH_LIST 8.6)

project(DiffusionModelC++ C CXX)

############################################################################################################
# Compile Options 
############################################################################################################
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE on)

# Set default build type if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

if (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
elseif (UNIX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif ()

############################################################################################################
# Environment Variables
############################################################################################################
set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(PROJECT_EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external")

############################################################################################################
# External Libraries
############################################################################################################
# OpenMP
find_package(OpenMP REQUIRED)

if (OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Torch
set(CAFFE2_USE_CUDNN 1)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# OpenCV
find_package(OpenCV REQUIRED)

# Spdlog
set(SPD_LOG_INCLUDE_DIR ${PROJECT_EXTERNAL_DIR}/spdlog/include)

# Pico JSON
set(PICO_JSON_INCLUDE_DIR ${PROJECT_EXTERNAL_DIR}/picojson)

############################################################################################################
# External libralies
############################################################################################################
set(
    EXTERNAL_INCLUDE_DIR
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${SPD_LOG_INCLUDE_DIR}
    ${PICO_JSON_INCLUDE_DIR}
)

set(
    PROJECT_LIBS
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

############################################################################################################
# Add subdirectory
############################################################################################################
add_subdirectory(src)
add_subdirectory(external)
add_subdirectory(tests)

############################################################################################################
# Message
############################################################################################################
message(STATUS "# =======================================================================================================")
message(STATUS "#  '${PROJECT_NAME}' build configuration ")
message(STATUS "# =======================================================================================================")
message(STATUS "#  [CMake]")
message(STATUS "#    CMake version                        : ${CMAKE_VERSION}")
message(STATUS "#    CMake build type                     : ${CMAKE_BUILD_TYPE}")
message(STATUS "#    CMAKE_CXX_FLAGS                      : ${CMAKE_CXX_FLAGS}")
message(STATUS "#    CMAKE_CXX_FLAGS_DEBUG                : ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "#    CMAKE_CXX_FLAGS_RELEASE              : ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "# ")
message(STATUS "#  [C/C++]")
message(STATUS "#    C   Compiler                         : ${CMAKE_C_COMPILER_ID} | ${CMAKE_C_COMPILER_VERSION} | ${CMAKE_C_COMPILER}")
message(STATUS "#    C++ Compiler                         : ${CMAKE_CXX_COMPILER_ID} | ${CMAKE_CXX_COMPILER_VERSION} | ${CMAKE_CXX_COMPILER}")
message(STATUS "#    CXX STANDARD                         : ${CMAKE_CXX_STANDARD}")
message(STATUS "# ")
message(STATUS "#  [CUDA]")
message(STATUS "#    CMAKE_CUDA_COMPILER                  : ${CMAKE_CUDA_COMPILER}")
message(STATUS "#    CMAKE_CUDA_ARCHITECTURES             : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "# ")
message(STATUS "#  [OpenMP]")
message(STATUS "#    OpenMP_C_FLAGS                       : ${OpenMP_C_FLAGS}")
message(STATUS "#    OpenMP_CXX_FLAGS                     : ${OpenMP_CXX_FLAGS}")
message(STATUS "# ")
message(STATUS "#  [Environment Variables]")
message(STATUS "#    PROJECT_INCLUDE_DIR                  : ${PROJECT_INCLUDE_DIR}")
message(STATUS "#    PROJECT_EXTERNAL_DIR                 : ${PROJECT_EXTERNAL_DIR}")
message(STATUS "# =======================================================================================================")