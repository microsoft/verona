Process-based sandbox experiment
================================

This experiment is attempting to build the base-line sandboxing mechanism for foreign code in Verona.
As a design principle, Verona does not permit unsafe code to run outside of a sandbox (with the exception of the small run-time library, which can be carefully audited and possibly replaced with formally verified code in the future).
The sandboxing abstractions in the language can be implemented in a variety of different ways, such as:

 - Process-based isolation.
 - MMU-based isolation with a non-process abstraction.
 - Software fault isolation (e.g. via a WebAssembly implementation).
 - Novel hardware features (e.g. CHERI)

Process-based isolation is attractive as a baseline because it does not require a modified OS and depends only on CPU features that have been on all mainstream CPUs for decades.

Terminology
-----------

The process-sandbox code uses the following terminology:

 - *Parent*: The trusted process that wishes to load one or more untrusted libraries.
 - *Child*: The process that runs the unprivileged library.
 - *Sandbox*: The child process and associated state, including the shared heap.
 - *Sandbox heap*: A region of memory shared between the *parent* and *child* from which both can dynamically allocate memory.
 - *Sandboxed library*: The interface that manages all of the state associated with a sandbox and exposes its functionality to the *parent*.
 - *Sandboxed function*: A function exposed from a *sandboxed library*.
 - *Callback function*: The converse of a *sandboxed function*, a function implemented in the parent and exposed for the child to call.
 - *Sandbox allocator*: The snmalloc allocators that run inside a sandbox and allocate memory from the *sandbox heap*.
 - *Boundary allocator*: The per-sandbox instance of snmalloc that allocates memory in the *sandbox heap* on behalf of the *parent*.
 - *OS sandbox*: The kernel-specific functionality used to restrict the rights of the *child*.
 - *library runner*: The program (the `library_runner` binary, compiled from [library_runner.cc](src/library_runner.cc)) that the child process runs.

High-level abstractions
-----------------------

The host process is assumed to be trusted and wishes to:

 - Load untrusted libraries into a sandbox.
 - Allocate memory accessible to the sandbox.
 - Access data structures in sandbox-owned memory.
 - Provide the library with access to external resources (e.g. files or sockets).
 - Invoke functions / methods within the sandbox.
 - Implement callbacks that the sandboxed code can invoke.

All memory in the sandbox's heap is mapped into both the host and child at the same address and so pointers to sandbox-heap memory can be used directly in the child and after a range check in the parent.

Security
--------

The sandboxing code is a critical part of the TCB for Verona and so security is one of the most important parts of the design.

### Threat model

The sandbox loads a library that is assumed to be untrusted and so the attacker is assumed to have arbitrary-code execution power within the sandbox.
The attacker is allowed to corrupt *any* memory owned by the sandbox, including the sandbox heap.
Similarly, the attacker is assumed to be able to modify memory in the sandbox heap concurrently with accesses from the parent.

The attacker must not be able to:

 - Directly modify any memory owned by the parent.
 - Cause the parent to modify any non-sandbox memory except as explicitly allowed via safe callbacks.
 - Cause the parent to stop executing (denial of service) in any thread, except for the time that the parent allows for a call into the child to execute.
 - Access any global namespace (filesystem, network, and so on) except as explicitly permitted by the parent.

### Unknown issues

This is experimental code and has not yet been audited for security.
As such, there are almost certainly security issues, possibly some that are intrinsic to the design.
**DO NOT USE THIS CODE IN PRODUCTION (yet)**

Design
------

The process sandbox code creates a child process and configures it to run in a low-privilege mode, with a shared memory region for the sandbox heap.
The child receives a small number of handles that allow it to communicate with the parent.

The core of the design provides a lightweight mechanism for allocating memory in the shared region.
Other abstractions are built on this fact to provide lightweight single-copy I/O (in some cases zero-copy, but typically a single copy is required to prevent time-of-check-to-time-of-use vulnerabilities).

Memory management
-----------------

For each sandbox instance, the parent creates two anonymous shared memory objects.
One object is large and will be used for the sandbox's heap.
The large object is mapped at the same address in both the parent and a child process, allowing intra-sandbox pointers to have the same values in both.
This is mapped read-write in both processes.

The smaller object is a single page, used for the snmalloc page map that covers the large region.
This is mapped read-write in the parent over the region of the pagemap that contains metadata for the large region.
It is mapped read-only in the child.
Updates to the pagemap are infrequent.

The parent has a boundary allocator associated with the sandbox that allows allocation in the sandbox's heap.
The parent allocator's message queue is in the shared region, but the rest of the state is not (though free lists are stored inline in objects and so must be untrusted).
When the boundary allocator in the parent needs to update the pagemap, it does so directly.
When an sandbox allocator in the child needs to update the parent, it communicates the update via a pipe to the parent.
The parent then validates the requested update and performs the write.

### Pagemap updates and chunk allocation

The back end of the snmalloc memory allocator communicates with a memory provider that is responsible for providing chunks of memory. 
This is on the slow path - once a chunk (typically 1 MiB) has been allocated to an allocator, memory from that chunk can be allocated very cheaply.
The memory provider resides in the parent because that is the only place where trusted code can run.

The child sends pagemap updates and memory provider requests to the parent via a socket.
Because all pagemap updates are done in the parent, allocators do not need to validate the pagemap values on read.

The socket is inherited from the parent in the file descriptor identified by `PageMapUpdates` (defined in [sandbox_fd_numbers.h](include/process_sandbox/sandbox_fd_numbers.h)).
These updates are sent via the trivial RPC protocol described in [host_service_calls.h](src/host_service_calls.h).
This is not a general protocol but is designed with the following constraints:

 - It must not require memory allocation because it is invoked during memory allocation.
 - It must not require any other dynamically allocated state because it is invoked during early bootstrapping in the child.
 - It does not need to support reentrancy and, in particular, should not because allocation should always make forward progress (possibly by failing if the shared heap address space is exhausted).

When the child starts, the libc bootstrapping code invokes `malloc`, which triggers snmalloc to bootstrap (via its slow-path initialisation) by mapping the shared memory region and invoking the RPC to request a chunk from which it can allocate.
Once this is done, it's possible for both the child and parent to allocate memory within the shared heap.

Note that, for this to be efficient in the current code, the OS must implement lazy commit so that allocating a large (e.g. 1GiB) shared memory region does not consume 1GiB of physical memory or swap unless it is actually used.
Memory is not used until the allocator has requested it from the memory provider, and so it would be possible to only commit pages in the shared region as requested (and even to provide each superslab as a separate shared memory object).

Starting the child process
--------------------------

On process creation, everything running in the child is trusted code.
The OS sandbox abstraction allows policies to be run before or after the `library_runner` process starts, but at the moment the security guarantees are roughly equivalent.

The child closes the file descriptors used for the shared mapping and opens the requested library.
The library is a shared library that provides the interfaces to the untrusted code and links directly to that code.
As part of the process of loading the library, the run-time linker will invoke global constructors and so after this point the child is not trusted.

The loaded library is expected to expose two functions:

 - `sandbox_init` is the equivalent of `main`, it is invoked after the library is loaded and all global constructors have run but before any exported functions are invoked.  This function takes no arguments and does not return a value.
 - `sandbox_call` is the function that is invoked whenever the parent calls a sandboxed function.

The `sandbox_call` function takes an integer and a pointer as arguments.
The integer is intended to contain the index of the exported function in a logical vtable for the library, the pointer points to a structure used to marshal the arguments and return values.

After the library runner has found the entry points into the library, it enters the runloop where it waits for the parent to call functions.

Calling sandbox functions
-------------------------

The call-return mechanism for invoking sandbox functions is richer than the host service call mechanism for the memory allocator.
It is free to depend on a working memory allocator, because it is used only after snmalloc's bootstrapping completes and snmalloc does not, itself, use this mechanism.
It also must support recursive calls with stack stitching.
At a high level, the parent must be able to call into the sandbox, which may then invoke callbacks, which then may invoke the sandbox, and so on, with a single logical call stack.

### Flow control for calls

When the parent thread invokes the child, it must sleep until the child either returns, or notifies it to invoke a callback.
The core building block for signalling between the two is a one-bit semaphore in the shared memory region.
This is used to implement a token-passing mechanism.
At almost any time, the child will be blocked on its semaphore waiting for the parent or a parent thread will be blocked on its semaphore waiting for the child to complete.

When a child starts, both it and the parent are running.
If the run-time linker does not natively support a sandboxing technology (e.g. seccomp-bpf and the glibc `ld-linux.so`) then the parent must be able to handle callbacks from the client to open shared libraries before it can call the first sandboxed function.
This causes some slightly complex logic for the initial rendezvous, which will probably be simplified in a future version.

The structure that defines the static (non-heap) part of the shared memory region is declared in [shared_memory_region.h](include/process_sandbox/shared_memory_region.h).
This has a field called `token` that contains a pair of one-bit semaphores and the stack depth of current callback.
The depth is incremented in the child before it invokes the parent and decremented in the parent when it returns.
This allows each side to sit in a modal runloop in the stack frame responsible for waiting for the return and handle deeper invocations by local recursion.
When a runloop is woken up in the child, it checks whether the depth has been decreased by the parent and, if so, returns, otherwise it handles a new invocation from the parent.
When a runloops is woken up in the parent, it checks whether the depth has been increased by the child and, if so, invokes the correct callback function, otherwise it returns.

The code for invoking the sandbox is in `sandbox::Library::send` in [libsandbox.cc](src/libsandbox.cc), the code for handling invocations in the child is in the `runloop` function in [library_runner.cc](src/library_runner.cc).

### Data movement for calls

The mechanism described in the last section transfers control from the parent to the child and back, increasing or decreasing the call depth, but it does not describe how parameters and return values are passed.
Recall that we can cheaply allocate memory in the shared heap from the parent or the child.
We take advantage of this for arguments, allocating a structure that contains the arguments and space for the return value and passing a pointer to it in a fixed location in the shared memory region.

For example, consider exporting a function such as this from a sandbox:

```C
int example(struct SomeStruct s);
```

The wrapper code (generated by the Verona compiler or by the C++ API) would generate a structure like this (names for illustration only, these would be opaque in the Verona implementation and are tuple elements and not structure fields in the C++ version):

```C
struct ExampleArgFrame
{
	int ret;
	struct SomeStruct s; 
};
```

The `sandbox_call` function would be passed a pointer to this when invoked with the index of this function.
It is then responsible for calling `example` with the argument from the structure and placing the return address inside the structure.
The Verona compiler will synthesize something roughly like this:

```C
void sandbox_call(int idx, void *args)
{
	switch (idx)
	{
		...
		case ExampleFunctionNumber: // Arbitray value, must be unique
		{
			struct ExampleArgFrame *argframe = args;
			args->ret = example(args->s);
		}
	}
}
```

The functions in [cxxsandbox.h](include/process_sandbox/cxxsandbox.h) provide templates for generating this code directly from C++ function declarations.
Note that the C++ APIs provide the same abstractions for sandbox code invocation that the Verona compiler is expected to use, but the C++ type system lacks viewpoint adaptation and so cannot help you avoid accidentally following pointers that the attacker is able to manipulate.

Currently, the in-memory RPC mechanism used to invoke methods in the child is very high latency.
This may not be a problem for Verona, where foreign calls are likely to be wrapped in `when` clauses, which can batch multiple operations within the library.
The asynchronous operation of `when` clauses hides latency, avoiding the blocking operations in the C++ proof-of-concept.
This overhead could be significantly reduced on an OS that supported Spring / Solaris Doors.

### Callbacks and system call emulation

Callbacks are registered with the sandboxed library and are assigned a number.
Currently, the callback mechanism sends the equivalent of the two arguments that are passed to the `sandbox_call` function over a socket, rather than using an in-memory transport.
This is because you can't send a file descriptor over a UNIX domain socket without also sending some data and callbacks need to be able to both accept and return file descriptors.
At some point, the two mechanisms will be unified.
Linux, for example, provides Windows-like system calls for getting and inserting file descriptors in a child process as of version 5.10, which could be used to fetch and return file descriptors if required rather than requiring the socket.

The same callback mechanism is used for system call emulation.
When the child wishes to invoke a system call that is not allowed (for example, `open`, which would grant access to the entire filesystem if permitted), this is handled by a callback that takes the arguments and returns either an error value or a file descriptor.
Currently, only a small handful of system calls (those required for glibc's `ld-linux.so` to load a library) are proxied but this will grow over time.

OS sandboxing mechanisms
------------------------

The OS sandbox abstraction layer will almost certainly change over time.
It currently supports Capsicum and seccomp-bpf but could potentially support the macOS sandbox framework, OpenBSD's pledge, and so on.

Capsicum provides two small modifications to the traditional POSIX system call API:

 - File descriptors have fine-grained rights that can be removed at any time by an explicit call (but cannot be re-added).
 - A process in Capsicum mode has no access to any global namespace and so cannot issue system calls such as `open` or `bind` at all and can only issue relative calls such as `openat` if it has a file descriptor with the relevant permissions.

Conveniently, this is *precisely* the policy that we want for sandboxing libraries.
The library running in the child can do whatever it wants with its own heap, but it can only read or write files, sockets, or IPC primitives that the parent explicitly delegates it access to.
Enforcement of a Capsicum sandbox is very cheap and there is no performance penalty for running in the sandboxed mode.

The seccomp-bpf is far less useful for this purpose.
The model for seccomp-bpf is that userspace can provide a filter program that runs on system call entry that determines whether the system call may proceed.
This has several limitations compared to Capsicum:

 - There is no way of allowing `openat` in the safe cases, so `openat` must be emulated.
 - The system call filters can make decisions based on their arguments only in the most limited sense:
   - They cannot attach metadata to a file descriptor and make decisions based on the specific file descriptor.
   - They cannot (for security reasons) read userspace memory and so cannot inspect any pointer arguments (e.g. paths, flags for `clone3` and so on).
   - The `libseccomp` library does not expose interfaces for comparing two arguments and so you cannot write filters of the form `base + length > sandbox_heap_start` without opting out of `libseccomp` entirely and writing raw eBPF scripts.

The seccomp-bpf policy divides system calls into four categories:

 - Safe to use in general (e.g. `read` / `write`)
 - Process-related state, safe to use only if the process ID refers to this process, blocked otherwise.
 - Unsafe to use in general but possibly safe in some cases and so emulated with a callback where possible (e.g. `openat` which can be transformed into a callback that may or may not open the file for you, depending on user policy).
 - Unsafe to use in any case (e.g. `settimeofday`), includes intrinsically privileged operations that you should not try to use in a sandbox.

### System-call interposition

In the common case, POSIX software invokes system calls via libc wrappers.
ELF linkage allows us to preempt those symbols by providing implementations in `library_runner` that will be use in preference to the ones in libc.
This makes it possible to intercept calls that would normally result in a call that the operating system would block and issue a callback instead.
Some software; however, issues system calls directly.
In particular, glibc's `ld-linux.so`, which is required for loading the untrusted library *after* the sandbox policy has been applied on GNU/Linux systems, issues system calls directly to open the file.

Both of the currently supported sandboxing mechanisms provide a mechanism for delivering a signal if the sandbox policy blocks a call.
The `library_runner` process registers a handler for this system call and attempts a last-resort emulation for the system calls that we can transform into callbacks.
This extracts the system call number and arguments from the signal frame (which include a complete register dump at the time of the signal) and then re-inserts the results before the signal returns.

Signal delivery is generally a very slow path in operating systems and so this mechanism is definitely not the desired way of issuing system calls.
It provides a fallback for maximising compatibility but in the common case this mechanism should not be used.
All of the complex code for handling this case runs inside the sandbox and so a bug in it would not grant the attacker any power that they do not have within our threat model.

Portability
-----------

The current code is expected to be easily portable to any POSIX system that has a process sandboxing framework.
The core abstractions provided should be portable to Windows using the Isolated User Mode APIs, though interposing on Windows system calls for privilege elevation may not be possible and there are still a number of questions about how this can be implemented.

The platform abstractions are all in the [platform](include/process_sandbox/platform/) directory.
This currently provides several abstractions that have generic POSIX implementations:

 - Child processes, implemented via `vfork`, with a `pdfork` implementation available if supported.
 - Socket pairs with the ability to send and receive file descriptors and data.
   Windows provides separate mechanisms for sending messages and sending handles but these should be possible to wrap in the same abstraction.
 - Shared memory, implemented with generic POSIX shared memory objects and `mmap`.
   This is more efficient if the OS can guarantee strongly aligned allocations and if it provides native support for anonymous memory objects.

The remaining abstractions require per-OS (or OS-family) support.
These are:

 - A poller, which can wait for data on a set of socket connections and notice when the remote end is closed.
   This is implemented with `epoll` (Linux) and `kqueue` (pretty much everything that's not Linux).
   The POSIX `select` and `poll` calls are not quite adequate because they do not portably allow detecting when the remote end of a socket or pipe has been closed.
 - A one-bit semaphore that can exist in memory.
   This is *probably* safe to implement with a pthread mutex and condition variable but it is not possible to portably reason about the security implications of doing so.
   There are `futex` (Linux, OpenBSD) and `umtx` (FreeBSD) implementations.
   The `futex`-based implementation will require some small tweaks to run on OpenBSD.
   This should be possible to implement on any platform with a mutex primitive that does not store any critical data in userspace other than the lock word.
 - System call frame introspection routines.
   These are both platform- and architecture-dependent.
   They are required to pull the system-call number arguments out of a trap frame delivered to a signal and re-inject the return values.
   This is purely a compatibility feature and is not required if running libraries that only invoke system calls via interposable functions.
 - Finally, and most importantly, the OS sandboxing policy code.
   Capsicum (FreeBSD) and seccomp-bpf (Linux) implementations are provided.
   The Capsicum implementation is the simplest, because the kernel provide exactly the policy required (no direct access to the global namespace, restrictions on the operations permitted on delegated file descriptors.
   The seccomp-bpf policy blocks a number of things that should be safe most of the time and requires callbacks to implement them.

Code layout
-----------

 - [include/process_sandbox](include/process_sandbox) contains the public interfaces.
   - [cxxsandbox.h](include/process_sandbox/cxxsandbox.h) contains the C++ API, which is primarily used for testing.
   - [filetree.h](include/process_sandbox/filetree.h) describes the interface for exporting a virtual file tree to the child.
     This is accessed from the `sandbox::Library` class.
   - [helpers.h](include/process_sandbox/helpers.h) provides some helpers for managing C memory, extracting argument types from functions, and so on.
   - [platform/](include/process_sandbox/platform) contains the platform abstraction code.
   - [callbacks.h](include/process_sandbox/callbacks.h) describes the callback mechanism.
   - [sandbox_fd_numbers.h](include/process_sandbox/sandbox_fd_numbers.h) contains the file descriptors that are set on child-process creation.
   - [sandbox.h](include/process_sandbox/sandbox.h) contains the definition of the sandbox library interface.
   - [shared_memory_region.h](include/process_sandbox/shared_memory_region.h) defines the part of the shared memory region, not including the heap.
 - [src](src) contains the source files
   - [child_malloc.h](src/child_malloc.h) contains the interfaces for the parts that specialise snmalloc for use in the child.
   - [host_service_calls.h](src/host_service_calls.h) describes the IPC mechanism used for the snmalloc to request memory and update the sandbox.
   - [library_runner.cc](src/library_runner.cc) is the program that runs inside the sandbox that loads the library and manages communication with the parent.
   - [libsandbox.cc](src/libsandbox.cc) is the library that manages sandbox lifecycle.
 - [tests](tests) contains tests.
   Some are stand-alone unit tests, the files that start `sandbox-` and `sandboxlib-` are the parent / child parts of tests that use the complete sandboxing framework.
