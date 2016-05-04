add_definitions(-std=c++11)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS serialization chrono timer system)

# mlpack test executable.
# mlpack test executable.
include_directories(${Boost_INCLUDE_DIRS})
add_executable(mlpack_test
  serial_code.cpp
)
# Link dependencies of test executable.
target_link_libraries(mlpack_test
  mlpack
  ${Boost_LIBRARIES}
)

add_executable(par_test
  parallel_code.cpp
)
# Link dependencies of test executable.
target_link_libraries(par_test
  mlpack
  ${Boost_LIBRARIES}
)
