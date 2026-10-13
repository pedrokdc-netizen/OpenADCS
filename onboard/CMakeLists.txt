cmake_minimum_required(VERSION 3.16)
project(openadcs_onboard VERSION 0.1.0 LANGUAGES C)

add_library(openadcs_processing
    src/sun_sensor.c
    src/gyro.c
    src/adcs.c
    src/gps.c
)
target_include_directories(openadcs_processing PUBLIC include)
set_target_properties(openadcs_processing PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE YES
)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(openadcs_processing PRIVATE OPENADCS_SHARED)
endif()
if(NOT MSVC)
    target_compile_options(openadcs_processing PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(openadcs_processing PUBLIC m)
endif()
