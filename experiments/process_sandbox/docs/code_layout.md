# Code layout

 - [include/process_sandbox](../include/process_sandbox) contains the public interfaces.
   - [cxxsandbox.h](../include/process_sandbox/cxxsandbox.h) contains the C++ API, which is primarily used for testing.
   - [filetree.h](../include/process_sandbox/filetree.h) describes the interface for exporting a virtual file tree to the child.
     This is accessed from the `sandbox::Library` class.
   - [helpers.h](../include/process_sandbox/helpers.h) provides some helpers for managing C memory, extracting argument types from functions, and so on.
   - [platform/](../include/process_sandbox/platform) contains the platform abstraction code.
   - [callbacks.h](../include/process_sandbox/callbacks.h) describes the callback mechanism.
   - [sandbox_fd_numbers.h](../include/process_sandbox/sandbox_fd_numbers.h) contains the file descriptors that are set on child-process creation.
   - [sandbox.h](../include/process_sandbox/sandbox.h) contains the definition of the sandbox library interface.
   - [shared_memory_region.h](../include/process_sandbox/shared_memory_region.h) defines the part of the shared memory region, not including the heap.
 - [src](src) contains the source files
   - [child_malloc.h](../src/child_malloc.h) contains the interfaces for the parts that specialise snmalloc for use in the child.
   - [host_service_calls.h](../src/host_service_calls.h) describes the IPC mechanism used for the snmalloc to request memory and update the sandbox.
   - [library_runner.cc](../src/library_runner.cc) is the program that runs inside the sandbox that loads the library and manages communication with the parent.
   - [libsandbox.cc](../src/libsandbox.cc) is the library that manages sandbox lifecycle.
 - [tests](../tests) contains tests.
   Some are stand-alone unit tests, the files that start `sandbox-` and `sandboxlib-` are the parent / child parts of tests that use the complete sandboxing framework.
