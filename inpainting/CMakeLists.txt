cmake_minimum_required(VERSION 2.8)
project(inpainting)

find_package(OpenCV REQUIRED)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

include(FindOpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# The version number.
set(inpainting_VERSION_MAJOR 0)
set(inpainting_VERSION_MINOR 0)

add_library(utils STATIC utils.cpp)
target_link_libraries(utils ${OpenCV_LIBS})

add_library(Inpainting STATIC Inpainting.cpp)
target_link_libraries(Inpainting ${OpenCV_LIBS})

add_executable(main main.cpp)
target_link_libraries(main Inpainting)
target_link_libraries(main ${OpenCV_LIBS})
