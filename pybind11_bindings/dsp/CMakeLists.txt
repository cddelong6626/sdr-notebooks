file(GLOB_RECURSE DSP_SOURCES src/*.cpp)
file(GLOB_RECURSE DSP_HEADERS include/**/*.hpp)

add_library(dsp STATIC ${DSP_SOURCES} ${DSP_HEADERS})

target_include_directories(dsp PUBLIC include)
