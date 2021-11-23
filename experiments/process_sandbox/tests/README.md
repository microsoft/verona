# process_sandbox Tests

Tests
------

This folder contains 17 tests, 8 unit and regression tests, and 9 end-to-end sandbox tests.

## End-to-End sandbox tests

* [test-sandbox-basic](sandbox-basic.cc): Creates 3 sandboxes wrapping around a small library that exposes a function returning the sum of two `int`.
  Each sandbox is called twice.

* [test-sandbox-callback-basic](sandbox-callback-basic.cc): Creates a simple sandbox and registers a callback function that returns an `int`.
   The sandbox library simply invokes the callback and asserts it gets the correct value (`42`). 

* [test-sandbox-callback-recursive](sandbox-callback-recursive.cc): Creates a single sandbox that recursively invokes 6 nested levels of callbacks, i.e., a callback triggers a call into the sandbox that itself invokes a callback etc. 

* [test-sandbox-crash](sandbox-crash.cc): Creates 3 sandboxes meant to crash (calls `abort`).
  The crash triggers a `runtime_error` exception caught in the parent process. 

* [test-sandbox-fake-open](sandbox-fake-open.cc): Creates a shared memory file and a sandbox. The sandbox opens the file in two different ways: 1) with a raw system call, and 2) with the `open` function that performs a callback to the host. 

* [test-sandbox-modify-pagemap](sandbox-modify-pagemap.cc): Creates a sandbox attempting to modify the access rights of the pagemap using `mprotect`.
  The `mprotect` call should fail with a `permission denied`.

* [test-sandbox-zlib](sandbox-zlib.cc): Uses `zlib` to `deflate`, i.e., compress, the program's file. This test compares the output produced by a sandboxed and an unsandboxed version of zlib. 

* [test-sandbox-network](sandbox-network.cc): Example of how network policies can be used to restrict the sandbox's ability to perform network operations. 
This test creates a server sandbox and gradually enables network operations (1) `getaddrinfo`, (2) `bind`ing to a socket, (3) the ability to receive messages on the socket from the parent, and (4) the ability for asandboxed client to `connect` to the server. 

* [test-sandbox-curl](sandbox-curl.cc): 
This test demonstrates how users can define their own policies for network access. 
It creates a sandbox that wraps the `curl` library.
The sandbox is only allowed to perform lookups for and connect to `http://example.com`, i.e., its relies on custom policies for `getaddrinfo` and `connect`.
