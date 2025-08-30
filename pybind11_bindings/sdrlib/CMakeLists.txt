# Build sdrlib C++ library
file(GLOB_RECURSE SDRLIB_SOURCES src/*.cpp)
file(GLOB_RECURSE SDRLIB_HEADERS include/**/*.hpp)

add_library(sdrlib STATIC ${SDRLIB_SOURCES} ${SDRLIB_HEADERS})

target_include_directories(sdrlib PUBLIC include)

# Include KFR library
find_package(KFR CONFIG REQUIRED)
target_link_libraries(sdrlib PRIVATE kfr kfr_dsp)
