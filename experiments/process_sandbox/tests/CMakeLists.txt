project(snadbox-tests C CXX)
set(UNIT_TESTS
	child_process
	onebitsem-basic
	onebitsem-child
	path
	poller
	shm
	socketpair
	socketpair-sendhandle
	)

set(SANDBOX_TESTS
	basic
	crash
	fake-open
	callback-basic
	callback-recursive
	modify-pagemap
	network
	zlib
	)

find_package(Threads REQUIRED)

find_package(CURL)
if (${CURL_FOUND})
	list(APPEND SANDBOX_TESTS curl)
endif()

function(setup_sandbox_test_properties TEST_NAME)
	set_target_properties(${TEST_NAME} PROPERTIES
		INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include/;${CMAKE_SOURCE_DIR}/../../external/snmalloc/src"
	)
	target_link_libraries(${TEST_NAME} Threads::Threads)
	target_link_libraries(${TEST_NAME} fmt)
	if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		target_link_libraries(${TEST_NAME} rt)
	endif ()
endfunction()

foreach(TEST_NAME ${UNIT_TESTS})
	set(TEST_SRC "${TEST_NAME}.cc")
	set(TEST_NAME "test-${TEST_NAME}")
	add_executable(${TEST_NAME} ${TEST_SRC})
	setup_sandbox_test_properties(${TEST_NAME})
	add_test(${TEST_NAME} ${TEST_NAME})
endforeach()

foreach(TEST_NAME ${SANDBOX_TESTS})
	set(TEST_EXE_SRC "sandbox-${TEST_NAME}.cc")
	set(TEST_LIB_SRC "sandboxlib-${TEST_NAME}.cc")
	set(TEST_LIB_NAME "sandboxed-${TEST_NAME}")
	set(TEST_NAME "test-sandbox-${TEST_NAME}")
	add_executable(${TEST_NAME} ${TEST_EXE_SRC})
	add_library(${TEST_LIB_NAME} SHARED ${TEST_LIB_SRC})
	setup_sandbox_test_properties(${TEST_NAME})
	setup_sandbox_test_properties(${TEST_LIB_NAME})
	target_link_libraries(${TEST_NAME} sandbox z)
	target_link_libraries(${TEST_LIB_NAME} z)
	set_target_properties(${TEST_LIB_NAME} PROPERTIES PREFIX "")
	add_test(${TEST_NAME} ${TEST_NAME})
	target_compile_definitions(${TEST_NAME} PRIVATE SANDBOX_LIBRARY=\"${CMAKE_CURRENT_BINARY_DIR}/${TEST_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}\")
endforeach()

if (${CURL_FOUND})
	target_link_libraries(sandboxed-curl CURL::libcurl)
endif()
