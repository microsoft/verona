cmake_minimum_required(VERSION 3.8)
project(sandbox C CXX)

INCLUDE (CheckCXXSourceCompiles)

set(CHILD_SOURCES src/library_runner.cc)
set(LIBSANDBOX_SOURCES src/libsandbox.cc)

include_directories(AFTER "${CMAKE_SOURCE_DIR}/../../external/snmalloc/src")
include_directories(BEFORE "${CMAKE_SOURCE_DIR}/include")
set(CMAKE_CXX_STANDARD 17)


if (MSVC)
else ()
	add_compile_options(-mcx16 -Wall -Wextra -Werror "$<$<CONFIG:DEBUG>:-fno-inline>")
	# Disable GCC warnings that give a large number of false positives.
	if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
		add_compile_options(-Wno-attributes -Wno-cast-function-type)
	endif()
endif()

set(DEFAULT_KQUEUE true)
set(DEFAULT_PROCDESC false)
set(DEFAULT_CAPSICUM false)
set(DEFAULT_SECCOMP false)

if (${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
	set(DEFAULT_PROCDESC true)
	set(DEFAULT_CAPSICUM true)
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	set(DEFAULT_KQUEUE false)
	set(DEFAULT_SECCOMP true)
endif()

set(ENABLE_KQUEUE ${DEFAULT_KQUEUE} CACHE BOOL "Use kqueue for monitoring file descriptors")
set(ENABLE_PROCDESC ${DEFAULT_PROCDESC} CACHE BOOL "Use process descriptors and kqueue for creating the child process")
set(ENABLE_CAPSICUM ${DEFAULT_CAPSICUM} CACHE BOOL "Use Capsicum to restrict child process privileges")
set(ENABLE_SECCOMP ${DEFAULT_SECCOMP} CACHE BOOL "Use seccomp-pbf to restrict child process privileges")

if (${ENABLE_KQUEUE})
	message(STATUS "Using kqueue for polling")
	add_definitions(-DUSE_KQUEUE)
endif()
if (${ENABLE_PROCDESC})
	message(STATUS "Using process descriptors for monitoring child processes")
	add_definitions(-DUSE_KQUEUE_PROCDESC)
endif()
if (${ENABLE_CAPSICUM})
	message(STATUS "Using Capsicum for sandboxing")
	add_definitions(-DUSE_CAPSICUM)
endif()


add_library(sandbox SHARED ${LIBSANDBOX_SOURCES})
add_executable(library_runner ${CHILD_SOURCES})

find_package(Threads REQUIRED)
target_link_libraries(library_runner Threads::Threads)
target_link_libraries(sandbox Threads::Threads)

target_link_libraries(library_runner fmt)
target_link_libraries(sandbox fmt)
# The library runner needs to export the upcall function and its system call
# interposition functions to the loaded library.
target_link_options(library_runner PRIVATE -Wl,-export-dynamic)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	target_link_libraries(library_runner -ldl -lbsd)
	target_link_libraries(sandbox -ldl -lbsd -lrt)
endif()

if (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
	target_link_libraries(library_runner -latomic)
	target_link_libraries(sandbox -latomic)
endif()

if (${ENABLE_SECCOMP})
	message(STATUS "Using seccomp-bpf for sandboxing")
	target_link_libraries(library_runner -lseccomp)
endif()


enable_testing()
add_subdirectory(tests)

