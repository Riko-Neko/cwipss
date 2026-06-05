cmake_minimum_required(VERSION 3.18)

project(cwipss_native LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_pelt_ext MODULE src/cwipss/native/pelt_ext.cpp)
target_compile_features(_pelt_ext PRIVATE cxx_std_17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  target_compile_options(_pelt_ext PRIVATE /O2)
else()
  target_compile_options(_pelt_ext PRIVATE -O3)
endif()

set_target_properties(
  _pelt_ext
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/cwipss"
)

install(TARGETS _pelt_ext LIBRARY DESTINATION cwipss)
