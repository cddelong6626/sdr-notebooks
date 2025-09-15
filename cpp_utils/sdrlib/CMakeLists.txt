
# Build sdrlib C++ library
file(GLOB_RECURSE SDRLIB_SOURCES src/*.cpp)
file(GLOB_RECURSE SDRLIB_HEADERS include/**/*.hpp)

add_library(sdrlib STATIC ${SDRLIB_SOURCES} ${SDRLIB_HEADERS})

target_include_directories(sdrlib PUBLIC include)

# Include libraries
find_package(GTest CONFIG REQUIRED)
find_package(KFR CONFIG REQUIRED)

target_link_libraries(sdrlib PRIVATE kfr kfr_dsp)

# Unit testing
file(GLOB_RECURSE SDRLIB_TESTS tests/*.cpp)
add_executable(sdrlib_tests ${SDRLIB_TESTS})
target_link_libraries(sdrlib_tests PRIVATE sdrlib GTest::gtest GTest::gtest_main kfr kfr_dsp)

add_test(
    NAME sdrlib_tests
    COMMAND sdrlib_tests --gtest_color=yes
)