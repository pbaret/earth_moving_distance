cmake_minimum_required(VERSION 2.8.11)

project(emd)

ADD_DEFINITIONS ( -O3 -Wall -Wextra -Wno-unused -g -lboost_system)

#fin opencv package
find_package( OpenCV REQUIRED )

# SOURCE CODE :
set ( SOURCES
main.cpp
)

set ( HEADERS
)

# Tell CMake to create the helloworld executable
add_executable(emd ${SOURCES} ${HEADERS})

target_link_libraries(emd ${OpenCV_LIBS})    # Use OpenCV

