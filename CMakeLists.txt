cmake_minimum_required(VERSION 3.12)
project (torch-js)

# Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT MSVC)
  set(CMAKE_C_STANDARD 11)
endif()

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

add_subdirectory(decode_image)

include_directories(${CMAKE_JS_INC})
file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC})
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/node_modules/node-addon-api")

target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} -Wl,--no-as-needed decode_image)
target_link_libraries(${PROJECT_NAME} -Wl,--no-as-needed TorchVision::TorchVision)

add_definitions( -Wall -O2 -fexceptions )
