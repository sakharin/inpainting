cmake_minimum_required(VERSION 2.8)
project(inpainting)
find_package(OpenCV REQUIRED)

# The version number.
set(inpainting_VERSION_MAJOR 0)
set(inpainting_VERSION_MINOR 0)

add_library(utils STATIC utils.cpp)
target_link_libraries(utils ${OpenCV_LIBS})

add_executable(main main.cpp)
target_link_libraries(main utils)
target_link_libraries(main ${OpenCV_LIBS})
