cmake_minimum_required(VERSION 3.16)
project(tdoa_inverse_solver VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests" ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(tdoa_solver 
    src/main.cpp 
    src/solver.cpp 
    src/types.cpp
)

target_link_libraries(tdoa_solver 
    Eigen3::Eigen
)

if(BUILD_TESTS)
    enable_testing()
    add_executable(tdoa_tests 
        tests/test_solver.cpp 
        tests/test_gradient.cpp
    )
    target_link_libraries(tdoa_tests 
        Eigen3::Eigen
    )
    add_test(NAME solver_tests COMMAND tdoa_tests)
endif()
