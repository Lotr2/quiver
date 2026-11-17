include(GoogleTest)

add_executable(quiver_tests
  test_distance.cpp
  test_hnsw.cpp
  test_wal.cpp
  test_filter.cpp
  test_collection.cpp)
target_link_libraries(quiver_tests PRIVATE quiver::core GTest::gtest_main)
quiver_set_options(quiver_tests)
gtest_discover_tests(quiver_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 300)

if(TARGET quiver_api)
  target_sources(quiver_tests PRIVATE test_api.cpp)
  target_link_libraries(quiver_tests PRIVATE quiver::api)
endif()
