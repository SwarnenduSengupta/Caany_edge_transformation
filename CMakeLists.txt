cmake_minimum_required(VERSION 3.8)
project(canny_and_contour)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)
add_executable(canny_and_contour ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( canny_and_contour ${OpenCV_LIBS} )