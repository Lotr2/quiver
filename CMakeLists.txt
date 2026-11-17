cmake_minimum_required(VERSION 3.20)

project(quiver
  VERSION 0.1.0
  DESCRIPTION "An embeddable vector database with HNSW search, a write-ahead log and a REST API"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(QUIVER_BUILD_TESTS  "Build the unit and integration tests" ON)
option(QUIVER_BUILD_SERVER "Build the HTTP server"                 ON)
option(QUIVER_NATIVE       "Optimise for the build machine (-march=native)" OFF)
set(QUIVER_SANITIZE "" CACHE STRING "Sanitizers to enable, e.g. address,undefined or thread")
option(QUIVER_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

include(cmake/Dependencies.cmake)
include(cmake/CompilerOptions.cmake)

# ---------------------------------------------------------------------------
# Core library: distance kernels, HNSW index, WAL, collections
# ---------------------------------------------------------------------------
add_library(quiver_core
  src/distance.cpp
  src/hnsw_index.cpp
  src/flat_search.cpp
  src/crc32.cpp
  src/wal.cpp
  src/filter.cpp
  src/collection.cpp
  src/database.cpp)
add_library(quiver::core ALIAS quiver_core)
target_include_directories(quiver_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(quiver_core PUBLIC nlohmann_json::nlohmann_json Threads::Threads)
quiver_set_options(quiver_core)

# ---------------------------------------------------------------------------
# HTTP API (a library so the tests can drive it in-process) and the server
# ---------------------------------------------------------------------------
if(QUIVER_BUILD_SERVER)
  add_library(quiver_api server/http_api.cpp server/metrics.cpp)
  add_library(quiver::api ALIAS quiver_api)
  target_include_directories(quiver_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/server)
  target_link_libraries(quiver_api PUBLIC quiver::core httplib::httplib)
  quiver_set_options(quiver_api)

  add_executable(quiver-server server/main.cpp)
  target_link_libraries(quiver-server PRIVATE quiver::api)
  quiver_set_options(quiver-server)
  install(TARGETS quiver-server RUNTIME DESTINATION bin)
endif()

if(QUIVER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
