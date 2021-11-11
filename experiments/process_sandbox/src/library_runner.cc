// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "child_malloc.h"
#include "host_service_calls.h"
#include "process_sandbox/callbacks.h"
#include "process_sandbox/helpers.h"
#include "process_sandbox/platform/platform.h"
#include "process_sandbox/sandbox.h"
#include "process_sandbox/shared_memory_region.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <ucontext.h>
#include <unistd.h>

#ifndef MAP_FIXED_NOREPLACE
#  ifdef MAP_EXCL
#    define MAP_FIXED_NOREPLACE MAP_FIXED | MAP_EXCL
#  else
#    define MAP_FIXED_NOREPLACE MAP_FIXED
#  endif
#endif

using address_t = snmalloc::Aal::address_t;

// A few small platform-specific tweaks that aren't yet worth adding to the
// platform abstraction layer.
#ifdef __FreeBSD__
// On FreeBSD, libc interposes on some system calls and does so in a way that
// causes them to segfault if they are invoked before libc is fully
// initialised.  We must instead call the raw system call versions.
extern "C" ssize_t __sys_write(int fd, const void* buf, size_t nbytes);
extern "C" ssize_t __sys_read(int fd, void* buf, size_t nbytes);
#  define write __sys_write
#  define read __sys_read
#elif defined(__linux__)
namespace
{
  /**
   * Linux run-time linkers do not currently support fdlopen, but it can be
   * emulated with a wrapper that relies on procfs.  Each fd open in a
   * process exists as a file (typically a symlink) in /proc/{pid}/fd/{fd
   * number}, so we can open that.  This does depend on the running process
   * having access to its own procfs entries, which may be a problem for some
   * possible sandboxing approaches.
   */
  void* fdlopen(int fd, int flags)
  {
    char* str;
    if (asprintf(&str, "/proc/%d/fd/%d", (int)getpid(), fd) < 0)
    {
      DefaultPal::error("asprintf failed in fdlopen.");
    }
    void* ret = dlopen(str, flags);
    free(str);
    return ret;
  }
  typedef void (*dlfunc_t)(void);

  /**
   * It is undefined behaviour in C to cast from a `void*` to a function
   * pointer, but POSIX only provides a single function to get a pointer from a
   * library.  BSD systems provide `dlfunc` to avoid this but glibc does not,
   * so we provide our own.
   */
  dlfunc_t dlfunc(void* handle, const char* symbol)
  {
    return (dlfunc_t)dlsym(handle, symbol);
  }
}
#endif

extern "C"
{
  /**
   * The `environ` symbol is exported by libc, but not exposed in any header.
   *  We need to access this directly during bootstrap, when the libc functions
   *  that access it may not yet be ready.
   */
  extern char** environ;
}

using namespace sandbox;

namespace
{
  /**
   * Flag indicating that bootstrapping has finished.  Note that we cannot
   * create any threads until after malloc is set up and so this does not need
   * to be atomic: It is never modified after the second thread is created.
   */
  bool done_bootstrapping = false;

  /**
   * Bootstrap function.  Map the shared memory region and configure everything
   * needed for malloc.
   */
  SNMALLOC_SLOW_PATH
  void bootstrap();

  /**
   * The start of the shared memory region.  Passed as a command-line argument.
   */
  void* shared_memory_start = 0;

  /**
   * The end of the shared memory region.  Passed as a command-line argument.
   */
  void* shared_memory_end = 0;

  bool is_inside_shared_memory(const void* ptr, size_t size = 1)
  {
    return (ptr >= shared_memory_start) &&
      ((static_cast<const char*>(ptr) + size) < shared_memory_end);
  }

  /**
   * Pointer to the shared memory region.  This will be equal to
   * `shared_memory_start` and is simply a convenience to have a pointer of the
   * correct type.
   */
  SharedMemoryRegion* shared = nullptr;

  /**
   * Synchronous RPC call to the parent environment.  This sends a message to
   * the parent and waits for a response.  These calls should never return an
   * error and so this aborts the process if they do.
   *
   * This function is called during early bootstrapping and so cannot use any
   * libc features that either depend on library initialisation or which
   * allocate memory.
   */
  uintptr_t requestHostService(
    HostServiceCallID id,
    uintptr_t arg0,
    uintptr_t arg1 = 0,
    uintptr_t arg2 = 0,
    uintptr_t arg3 = 0)
  {
    static std::atomic_flag lock{0};
    FlagLock g(lock);
    HostServiceRequest req{id, {arg0, arg1, arg2, arg3}};
    auto written_bytes = write(PageMapUpdates, &req, sizeof(req));
    SANDBOX_INVARIANT(
      written_bytes == sizeof(req),
      "Wrote {} bytes, expected {}",
      written_bytes,
      sizeof(req));
    HostServiceResponse response;
    auto read_bytes = read(PageMapUpdates, &response, sizeof(response));
    SANDBOX_INVARIANT(
      read_bytes == sizeof(response),
      "Read {} bytes, expected {}",
      read_bytes,
      sizeof(response));

    if (response.error)
    {
      DefaultPal::error("Host returned an error.");
    }
    return response.ret;
  }

}

void SnmallocGlobals::ensure_init() noexcept
{
  if (!done_bootstrapping)
  {
    bootstrap();
  }
}

bool SnmallocGlobals::is_initialised()
{
  return done_bootstrapping;
}

namespace sandbox
{
  snmalloc::capptr::Chunk<void> SnmallocGlobals::reserve(size_t size)
  {
    SNMALLOC_ASSERT(size >= sizeof(void*));

    size = bits::align_up(size, sizeof(void*));

    size_t rsize = bits::next_pow2(size);

    snmalloc::capptr::Chunk<void> res{
      reinterpret_cast<void*>(requestHostService(
        MemoryProviderReserve, static_cast<uintptr_t>(rsize)))};

    return res;
  }

  void SnmallocGlobals::Pagemap::set_metaentry(
    LocalState*, snmalloc::address_t p, size_t size, snmalloc::MetaEntry t)
  {
    requestHostService(
      MetadataSet,
      static_cast<uintptr_t>(p),
      static_cast<uintptr_t>(size),
      reinterpret_cast<uintptr_t>(t.get_metaslab()),
      t.get_remote_and_sizeclass());
  }

  template<>
  snmalloc::capptr::Chunk<void>
  SnmallocGlobals::alloc_meta_data<snmalloc::Metaslab>(LocalState*, size_t size)
  {
    size = snmalloc::bits::next_pow2(size);
    return private_asm.reserve<true, SnmallocGlobals::PrivatePagemap>(
      nullptr, size);
  }

  template<>
  snmalloc::capptr::Chunk<void> SnmallocGlobals::alloc_meta_data<
    snmalloc::CoreAllocator<sandbox::SnmallocGlobals>>(LocalState*, size_t size)
  {
    size = snmalloc::bits::next_pow2(size);
    auto* result =
      reinterpret_cast<void*>(requestHostService(MemoryProviderReserve, size));
    return snmalloc::capptr::Chunk<void>{result};
  }

  std::pair<snmalloc::capptr::Chunk<void>, snmalloc::Metaslab*>
  SnmallocGlobals::alloc_chunk(
    SnmallocGlobals::LocalState*,
    size_t size,
    snmalloc::RemoteAllocator* remote,
    snmalloc::sizeclass_t sizeclass)
  {
    auto* ms = new (private_asm
                      .reserve<true, SnmallocGlobals::PrivatePagemap>(
                        nullptr, sizeof(snmalloc::Metaslab))
                      .unsafe_ptr()) snmalloc::Metaslab();
    auto result = requestHostService(
      AllocChunk,
      static_cast<uintptr_t>(size),
      reinterpret_cast<uintptr_t>(remote),
      static_cast<uintptr_t>(sizeclass),
      reinterpret_cast<uintptr_t>(ms));
    return {snmalloc::capptr::Chunk<void>{reinterpret_cast<void*>(result)}, ms};
  }
}

namespace
{
  /**
   * The function from the loaded library that provides the vtable dispatch
   * for functions that we invoke.
   */
  void (*sandbox_invoke)(int, void*);

  /**
   * The run loop.  Takes the public interface of this library (effectively,
   * the library's vtable) as an argument.  Exits when the callback depth
   * changes after executing the helper function.  This provides a nested
   * runloop abstraction similar to OpenStep's modal runloop. Each recursion
   * depth in a callback has its own runloop that handles recursive invocations
   * from the parent in response to the callback.
   */
  __attribute__((used)) void runloop(int callback_depth = 0)
  {
    int new_depth;
    do
    {
      // If `wait` spuriously fails, try again unless the parent has asked us
      // to exit.
      do
      {
        if (shared->should_exit)
        {
          exit(0);
        }
      } while (!shared->token.child.wait(INT_MAX));
      SANDBOX_DEBUG_INVARIANT(
        shared->token.is_child_executing,
        "Child is executing when the parent thinks is is not");
      int idx = shared->function_index;
      void* buf = shared->msg_buffer;
      shared->msg_buffer = nullptr;
      try
      {
        if ((buf != nullptr) && (sandbox_invoke != nullptr))
          sandbox_invoke(idx, buf);
      }
      catch (...)
      {
        // FIXME: Report error in some useful way.
        SANDBOX_INVARIANT(0, "Uncaught exception");
      }
      new_depth = shared->token.callback_depth;
      // Wake up the parent if it's expecting a wakeup for this callback depth.
      // The `callback` function has a wake but not a wait because it is using
      // the `wait` in this function, we need to ensure that we don't unbalance
      // the wakes and waits.
      if (new_depth == callback_depth)
      {
        shared->token.is_child_executing = false;
        shared->token.parent.wake();
      }
    } while (new_depth == callback_depth);
  }

  SNMALLOC_SLOW_PATH
  void bootstrap()
  {
    void* addr = nullptr;
    size_t length = 0;
    // Find the correct environment variables.  Note that libc is not fully
    // initialised when this is called and so we have to be very careful about
    // the libc function that we call.  We use the `environ` variable directly,
    // rather than `getenv`, which may allocate memory.
    //
    // The parent process provides the shared memory object in the file
    // descriptor with the number given by `SharedMemRegion` and the location
    // where it should be mapped in an environment variable.  The child has to
    // map this as the first step in bootstrapping (before most of libc
    // initialises itself) to get a working heap.
    for (char** e = environ; *e != nullptr; e++)
    {
      char* ev = *e;
      const char ev_name[] = "SANDBOX_LOCATION=";
      const size_t name_length = sizeof(ev_name) - 1;
      if (strncmp(ev_name, ev, name_length) == 0)
      {
        ev += name_length;
        char* end;
        addr = reinterpret_cast<void*>(strtoull(ev, &end, 16));
        SANDBOX_INVARIANT(
          end[0] == ':',
          "Expected ':' separator in environment variable, got '{}'",
          end[0]);
        length = strtoull(end + 1, nullptr, 16);
        break;
      }
    }
    // Abort if we weren't able to find the correct lengths.
    if ((addr == nullptr) || (length == 0))
    {
      DefaultPal::error("Unable to find memory location");
    }
    auto map_nocore = sandbox::platform::detail::map_nocore;

    // fprintf(stderr, "Child starting\n");
    // printf(
    //"Child trying to map fd %d at addr %p (0x%zx)\n", SharedMemRegion, addr,
    // length);
    void* ptr = mmap(
      addr,
      length,
      PROT_READ | PROT_WRITE,
      MAP_FIXED_NOREPLACE | MAP_SHARED | map_nocore,
      SharedMemRegion,
      0);

    // printf("%p\n", ptr);
    if (ptr == MAP_FAILED)
    {
      err(1, "Mapping shared heap failed");
    }

    shared = reinterpret_cast<SharedMemoryRegion*>(ptr);

    shared_memory_start = shared->start;
    shared_memory_end = shared->end;

    auto* base = static_cast<snmalloc::MetaEntry*>(mmap(
      nullptr,
      decltype(SnmallocGlobals::Pagemap::pagemap)::required_size(),
      PROT_READ,
      MAP_SHARED | map_nocore,
      PageMapPage,
      0));
    SANDBOX_INVARIANT(base != MAP_FAILED, "Mapping pagemap failed");
    SnmallocGlobals::Pagemap::pagemap.init(base);
    SnmallocGlobals::PrivatePagemap::pagemap.init();

    done_bootstrapping = true;
  }

  using sandbox::platform::Handle;
  using Socket = sandbox::platform::SocketPair::Socket;

  /**
   * The socket that is used for callbacks to the parent process.
   */
  Socket callbackSocket;

  /**
   * Invoke a callback.  This takes the kind of callback, the data to be sent,
   * and the file descriptor to send as arguments.  The file descriptor may be
   * -1, in which case the it is not sent.
   *
   *  The return value is the integer result of the callback and a `Handle` that
   *  is either invalid or the returned file descriptor.
   *
   *  This function should not be called directly, it should be invoked via the
   *  wrapper.
   */
  std::pair<uintptr_t, Handle>
  callback(sandbox::CallbackKind k, const void* buffer, size_t size, int fd)
  {
    Handle out_fd(fd);
    CallbackRequest req{k, size, reinterpret_cast<uintptr_t>(buffer)};
    callbackSocket.send(&req, sizeof(req), out_fd);
    out_fd.take();
    int depth = ++shared->token.callback_depth;
    shared->token.is_child_executing = false;
    shared->token.parent.wake();
    runloop(depth);
    Handle in_fd;
    CallbackResponse response;
    callbackSocket.receive(&response, sizeof(response), in_fd);
    return {response.response, std::move(in_fd)};
  }

  /**
   * Argument direction: for pointers, are they used to transfer data to or
   * from the callee?
   */
  enum ArgDirection
  {
    /**
     * The callee will read via this pointer.
     */
    In,
    /**
     * The callee will write via this pointer.
     */
    Out,
    /**
     * The callee will read and write via this pointer.
     */
    InOut
  };

  /**
   * Wrapper for pointer arguments.  Encapsulates a pointer to a `T` that is to
   * be passed to the parent.  Checks that the argument is inside the region
   * that is shared with the parent, allocating a new buffer if it is not.  If
   * this is an `Out` or `InOut` parameter, copies the result back.
   *
   * TODO: Handle arrays
   */
  template<typename T, ArgDirection Direction = In>
  class CallbackPtrArg
  {
    /**
     * The pointer captured by this object, `const`-qualified for parameters
     * that are not modified so that it can be constructed from
     * `const`-qualified parameters.
     */
    std::conditional_t<Direction == In, const T, T>* original;

    /**
     * An on-heap pointer for a copy of the pointee, allocated in the shared
     * region.  If `original` is in the shared heap, this will be `nullptr`.
     */
    std::unique_ptr<T> copy;

  public:
    /**
     * Constructor, captures `orig` and also allocates a copy of it on the
     * heap.  The value of `orig` must not be modified for the lifetime of this
     * object.
     */
    CallbackPtrArg(decltype(original) orig) : original(orig)
    {
      if (!is_inside_shared_memory(orig))
      {
        copy = std::make_unique<T>(*orig);
      }
    }

    /**
     * Destructor.  If this is an output parameter and the address passed out
     * of the sandbox was to the copy, copy the result back.
     */
    ~CallbackPtrArg()
    {
      if constexpr (Direction != In)
      {
        if (copy)
        {
          *original = *copy;
        }
      }
    }

    /**
     * Returns a pointer to the version of the argument that should be used.
     */
    const T* get()
    {
      return copy ? copy.get() : original;
    }

    /**
     * Provide the address (either of the original object if possible or the
     * copy if necessary) for use in a message to the parent.
     */
    uintptr_t arg()
    {
      return reinterpret_cast<uintptr_t>(get());
    }

    /**
     * Implicit conversion to the type used to communicate outside of the
     * sandbox.
     */
    operator uintptr_t()
    {
      return arg();
    }
  };

  /**
   * String pointer argument.  Encapsulates a C string pointer and copies it to
   * the sandbox shared heap if it is not within the sandbox.
   */
  class CallbackCStrArg
  {
    /**
     * The original C string.
     */
    const char* original;

    /**
     * A copy of the string in shared memory (if necessary);
     */
    unique_c_ptr<char> copy;

  public:
    /**
     * Constructor.  Holds a reference to the C string and copies it if
     * necessary.
     */
    CallbackCStrArg(const char* orig) : original(orig)
    {
      if (!is_inside_shared_memory(orig))
      {
        copy.reset(strdup(orig));
      }
    }

    /**
     * Provide the address (either of the original object if possible or the
     * copy if necessary) for use in a message to the parent.
     */
    uintptr_t arg()
    {
      return reinterpret_cast<uintptr_t>(copy ? copy.get() : original);
    }

    /**
     * Implicit conversion to the type used to communicate outside of the
     * sandbox.
     */
    operator uintptr_t()
    {
      return arg();
    }
  };

  /**
   * Invoke a callback, of the specified kind, passing `data`.  The `data`
   * argument must point to the shared heap.
   *
   * If the optional `fd` parameter is passed, then this file descriptor
   * accompanies the callback.  This is used for calls such as `openat`.
   */
  template<typename T>
  std::pair<uintptr_t, Handle>
  callback(sandbox::CallbackKind k, T* data, int fd)
  {
    return callback(k, data, sizeof(T), fd);
  }

  /**
   * Wrapper template for callbacks that are not system calls.  The first
   * template parameter is a specialisation of `sandbox::SyscallArgs<>`, which
   * infers the types from the underlying system call types.  The remaining
   * template parameters are the types used to construct this structure.  The
   * arguments are then implicitly converted to these types to handle things
   * like copying structures into shared memory.  The last argument is an
   * optional file descriptor to accompany the message.
   */
  template<typename ArgsStruct, typename... Args>
  std::pair<uintptr_t, Handle> callback_helper(Args... args, int fd = -1)
  {
    typename ArgsStruct::rpc_type stack_args{args...};
    CallbackPtrArg<decltype(stack_args)> argstruct{&stack_args};
    return callback(ArgsStruct::kind, argstruct.get(), fd);
  }

  /**
   * Wrapper template for callbacks that are system calls.  The first template
   * parameter is a specialisation of `sandbox::SyscallArgs<>`, which infers
   * the types from the underlying system call types.  The remaining template
   * parameters are the types used to construct this structure.  The arguments
   * are then implicitly converted to these types to handle things like copying
   * structures into shared memory.  The last argument is an optional file
   * descriptor to accompany the message.
   */
  template<typename ArgsStruct, typename... Args>
  intptr_t syscall_callback_helper(Args... args, int fd = -1)
  {
    typename ArgsStruct::rpc_type stack_args{args...};
    CallbackPtrArg<decltype(stack_args)> argstruct{&stack_args};
    auto ret = callback(ArgsStruct::kind, argstruct.get(), fd);
    int result = static_cast<int>(ret.first);
    // POSIX defines system calls to all return a single value (and optionally
    // set `errno`).  Some of those integer returns are file descriptors.  When
    // the file descriptor is sent over the UNIX domain socket, its number will
    // change.  If we receive a file descriptor, we return the received value
    // instead.
    //
    // Each operating system has a different convention for returning `errno`
    // values.  FreeBSD uses the carry bit to differentiate between a valid and
    // error return, for an invalid return the return value is moved to `errno`
    // and replaced with -1.  Irrespective of the host system, we always use
    // the Linux convention in this layer.  Linux uses the sign: Negative
    // values returned from the system call are stored in `errno` and the
    // return value replaced by -1, non-negative values are returned directly.
    // This is handled by the system call wrappers, these functions mimic the
    // raw system call because they are invoked from both the wrappers and the
    // system-call emulation in the signal handler.
    if (ret.second.is_valid())
    {
      result = ret.second.take();
    }
    return result;
  }

  /**
   * Emulate the `access` system call by performing a callback to the parent.
   */
  intptr_t callback_access(const char* path, int mode)
  {
    return syscall_callback_helper<
      sandbox::SyscallArgs<Access>,
      CallbackCStrArg,
      int>(path, mode);
  }

  /**
   * Emulate the `stat` system call by performing a callback to the parent.
   */
  intptr_t callback_stat(const char* pathname, struct stat* statbuf)
  {
    return syscall_callback_helper<
      sandbox::SyscallArgs<Stat>,
      CallbackCStrArg,
      CallbackPtrArg<struct stat, Out>>(pathname, statbuf);
  }

  /**
   * Emulate the `open` system call by performing a callback to the parent.
   */
  int callback_open(const char* pathname, int flags, mode_t mode)
  {
    return syscall_callback_helper<
      sandbox::SyscallArgs<Open>,
      CallbackCStrArg,
      int,
      mode_t>(pathname, flags, mode);
  }

  /**
   * Emulate the `bind` or `connect` system call by performing a callback to
   * the parent.  These two functions have the same argument signature and so
   * the logic can be shared between them.
   */
  template<CallbackKind Op>
  int callback_bind_or_connect(int s, const sockaddr* addr, socklen_t addrlen)
  {
    // The second argument is annoying because, although the static type is
    // `sockaddr` it is really some subtype of `sockaddr`, in a language that
    // can't express subtype relationships.  This is typically something like
    // `sockaddr_in` or `sockaddr_in6` but is implemented as a variable-sized
    // tagged union where the first few fields uniquely identify the type.
    // The size must be passed as an extra parameter.  We therefore can't use
    // the templates that automatically ensure a parameter is in shared memory.
    unique_c_ptr<sockaddr> copy;
    if (is_inside_shared_memory(addr, addrlen))
    {
      copy.reset(static_cast<sockaddr*>(malloc(addrlen)));
      memcpy(copy.get(), addr, addrlen);
      addr = copy.get();
    }
    return syscall_callback_helper<
      sandbox::SyscallArgs<Op>,
      int,
      uintptr_t,
      socklen_t>(s, reinterpret_cast<uintptr_t>(addr), addrlen, s);
  }

  /**
   * The callback functions use the Linux convention for system call returns:
   * non-negative numbers indicate success, negative numbers indicate values
   * that should be stored in `errno`.  The `syscall_return` function unwraps
   * this into the return value from the POSIX function, setting `errno` if
   * appropriate.
   */
  int syscall_return(int x)
  {
    if (x < 0)
    {
      errno = -x;
      return -1;
    }
    return x;
  }

  /**
   * Emulate the `openat` system call by performing a callback to the parent.
   */
  int callback_openat(int dirfd, const char* pathname, int flags, mode_t mode)
  {
#ifdef __linux__
    // Special case for Linux's /proc/{pid}/fd filesystem.  This is necessary
    // for fdlopen on Linux.
    char buf[128];
    pid_t pid = getpid();
    sprintf(buf, "/proc/%d/fd/%%d", (int)pid);
    int fd = -1;
    if (sscanf(pathname, buf, &fd) == 1)
    {
      return dup(fd);
    }
#endif
    if (pathname == nullptr)
    {
      return -EINVAL;
    }
    if (pathname[0] == '/')
    {
      return callback_open(pathname, flags, mode);
    }
    return syscall_callback_helper<
      sandbox::SyscallArgs<OpenAt>,
      int,
      CallbackCStrArg,
      int,
      mode_t>(dirfd, pathname, flags, mode, dirfd);
  }

  using SyscallFrame = sandbox::platform::SyscallFrame;

  /**
   * Helper template that extracts arguments from a system call register dump
   * passed into a signal handler.  This can then be passed to `std::apply` to
   * pass the arguments to a function.
   */
  template<typename Tuple, int Arg = std::tuple_size_v<Tuple> - 1>
  __attribute__((always_inline)) void extract_args(Tuple& args, SyscallFrame& c)
  {
    std::get<Arg>(args) = c.get_arg<
      Arg,
      std::tuple_element_t<Arg, std::remove_reference_t<Tuple>>>();
    if constexpr (Arg > 0)
    {
      extract_args<Tuple, Arg - 1>(args, c);
    }
  }

  /**
   * Callback functions that emulate system calls (not the C wrappers, which
   * will also set `errno`).  These must be kept in the same order as the
   * `CallbackKind` enumeration.
   */
  constexpr std::tuple callbacks{&callback_open,
                                 &callback_stat,
                                 &callback_access,
                                 &callback_openat,
                                 &callback_bind_or_connect<Bind>,
                                 &callback_bind_or_connect<Connect>};

  /**
   * Dispatch to a callback given a system call frame.  This will call the
   * entry in the `callbacks` tuple that matches the system call number.
   */
  template<int Number = SyscallCallbackCount - 1>
  void syscall_callback(SyscallFrame& c) noexcept
  {
    static_assert(
      Number < SyscallCallbackCount, "System call callback out of range");
    if constexpr (Number >= FirstSyscall)
    {
      // If the system call matches the one for this template instantiation:
      if (
        SyscallFrame::syscall_number<static_cast<CallbackKind>(Number)>() ==
        c.get_syscall_number())
      {
        // Extract the arguments from the system call frame and call the
        // handler.
        typename sandbox::internal::signature<std::remove_pointer_t<
          std::remove_reference_t<decltype(std::get<Number>(callbacks))>>>::
          argument_type args;
        extract_args(args, c);
        int result = std::apply(std::get<Number>(callbacks), args);
        // Put the result back into the argument frame.
        if (result < 0)
        {
          c.set_error_return(-result);
        }
        else
        {
          c.set_success_return(result);
        }
      }
      else
      {
        syscall_callback<Number - 1>(c);
      }
    }
  };

  /**
   * Signal handler function.  For system calls that are emulated after a trap,
   * this extracts the arguments from the trap frame, calls the correct callback
   * function, and then injects the return address into the syscall frame.
   */
  void emulate(int, siginfo_t* info, ucontext_t* ctx) noexcept
  {
    SyscallFrame c(*info, *ctx);
    if (c.is_sandbox_policy_violation())
    {
      syscall_callback(c);
    }
  }

}

/**
 * POSIX `open` function, performs a callback to the host rather than a system
 * call.
 */
extern "C" int open(const char* pathname, int flags, ...)
{
  mode_t mode = 0;
  if ((flags & O_CREAT) == O_CREAT)
  {
    va_list ap;
    va_start(ap, flags);
    mode = va_arg(ap, int);
    va_end(ap);
  }
  return syscall_return(callback_open(pathname, flags, mode));
}

#ifndef USE_CAPSICUM
/**
 * POSIX `openat` function, performs a callback to the host rather than a system
 * call.  In a Capsicum world, this is safe to allow the untrusted process to
 * do directly, so we don't bother interposing here.
 *
 * TODO: We should still interpose on this with Capsicum to handle things like
 * `openat(-100, "some/path", O_WHATEVER)`
 */
extern "C" int openat(int dirfd, const char* pathname, int flags, ...)
{
  va_list ap;
  va_start(ap, flags);
  mode_t mode = va_arg(ap, int);
  va_end(ap);
  int ret = callback_openat(dirfd, pathname, flags, mode);
  if (ret < 0)
  {
    errno = -ret;
    return -1;
  }
  return ret;
}
#endif

/**
 * The `bind` system call.  Delegates to the parent process.
 */
extern "C" int bind(int s, const sockaddr* addr, socklen_t addrlen)
{
  return syscall_return(callback_bind_or_connect<Bind>(s, addr, addrlen));
}

/**
 * The `connect` system call.  Delegates to the parent process.
 */
extern "C" int connect(int s, const sockaddr* addr, socklen_t addrlen)
{
  return syscall_return(callback_bind_or_connect<Connect>(s, addr, addrlen));
}

/**
 * The `getaddrinfo` libc call.  Because this is not a system call, we don't
 * need a split of the `callback_getaddrinfo` that can be called from both here
 * and the signal handler.
 */
extern "C" int getaddrinfo(
  const char* hostname,
  const char* servname,
  const addrinfo* hints,
  addrinfo** res)
{
  addrinfo* localRes;
  auto ret = callback_helper<
    sandbox::SyscallArgs<GetAddrInfo>,
    CallbackCStrArg,
    CallbackCStrArg,
    CallbackPtrArg<addrinfo>,
    CallbackPtrArg<addrinfo*, Out>>(hostname, servname, hints, &localRes);

  *res = localRes;
  return static_cast<int>(ret.first);
}

extern "C" void freeaddrinfo(addrinfo* ai)
{
  free(ai);
}

/**
 * Exported function to allow the loaded code to invoke a callback to the
 * parent.
 */
int sandbox::invoke_user_callback(int idx, void* data, size_t size, int fd)
{
  unique_c_ptr<void> copy;
  if (!is_inside_shared_memory(data, size))
  {
    copy.reset(malloc(size));
    memcpy(copy.get(), data, size);
    data = copy.get();
  }
  auto ret = callback(static_cast<sandbox::CallbackKind>(idx), data, size, fd);
  int result = static_cast<int>(ret.first);
  if (ret.second.is_valid())
  {
    result = ret.second.take();
  }
  return result;
}

#ifdef __x86_64__
/**
 * x86-64 assembly version of the stack pivot.  Stores the old stack pointer on
 * the new stack and then calls `runloop`.
 */
__asm__(
  "\t.type   runloop_with_stack_pivot,@function\n"
  "runloop_with_stack_pivot:\n"
  "\tmov  %rsp, %rbx # Old stack pointer in rbx\n"
  "\tmov  %rdi, %rsp # Move to using the new stack pointer\n"
  "\tpush %rbx       # Old stack pointer on the new stack\n"
  "\tpush %rbx       # Fix stack alignment\n"
  "\txor  %rdi, %rdi # Depth = 0\n"
  "\tcallq _ZN12_GLOBAL__N_17runloopEi # call runloop()\n"
  "\tpop %rax\n      # Ignored\n"
  "\tpop %rsp\n      # Restore the old stack pointer\n"
  "\tret             # Return to the return address from the old stack\n"
  ".Lrunloop_with_stack_pivot_end:\n"
  "\t.size   runloop_with_stack_pivot, "
  ".Lrunloop_with_stack_pivot_end-runloop_with_stack_pivot\n");

extern "C" void runloop_with_stack_pivot(void*, size_t);
#else
/**
 * Portable version of the stack pivot.  Creates a new pthread with the
 * specified stack and then waits for it to finish.
 */
static void runloop_with_stack_pivot(void* stack, size_t stack_size)
{
  pthread_t runner;
  pthread_attr_t attrs;
  pthread_attr_init(&attrs);
  pthread_attr_setstack(&attrs, stack, stack_size);
  // Create the new thread on the provided stack
  pthread_create(
    &runner,
    &attrs,
    [](void*) -> void* {
      runloop();
      return nullptr;
    },
    nullptr);
  // Block until the thread has exited.
  void* unused;
  pthread_join(runner, &unused);
}
#endif

int main()
{
  sandbox::platform::Sandbox::apply_sandboxing_policy_postexec();
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = (void (*)(int, siginfo_t*, void*))emulate;
  sigaction(SyscallFrame::syscall_signal, &sa, nullptr);
  // Close the shared memory region file descriptor before we call untrusted
  // code.
  close(SharedMemRegion);
  close(PageMapPage);
  callbackSocket.reset(FDSocket);

#ifndef NDEBUG
  // Check that our bootstrapping actually did the right thing and that
  // allocated objects are in the shared region.
  auto check_is_in_shared_range = [](void* ptr) {
    SANDBOX_DEBUG_INVARIANT(
      is_inside_shared_memory(ptr),
      "Pointer {} is out of the sandbox range {}--{}",
      ptr,
      shared_memory_start,
      shared_memory_end);
  };
  void* obj = malloc(42);
  check_is_in_shared_range(obj);
  free(obj);
#endif

  // Load the library using the file descriptor that the parent opened.  This
  // allows a Capsicum sandbox to prevent any access to the global namespace.
  // It is hopefully possible to implement something similar with seccomp-bpf,
  // though this may require calling into the parent to request additional file
  // descriptors and proxying all open / openat calls.
  void* handle = fdlopen(MainLibrary, RTLD_GLOBAL | RTLD_LAZY);
  if (handle == nullptr)
  {
    return 1;
  }

  // Find the library initialisation function.  This function will generate the
  // vtable.
  auto sandbox_init =
    reinterpret_cast<void (*)()>(dlfunc(handle, "sandbox_init"));
  if (sandbox_init == nullptr)
  {
    fprintf(stderr, "dlfunc failed: %s\n", dlerror());
    return 1;
  }
  // Set up the sandbox
  sandbox_init();
  sandbox_invoke =
    reinterpret_cast<decltype(sandbox_invoke)>(dlfunc(handle, "sandbox_call"));
  SANDBOX_INVARIANT(
    sandbox_invoke, "Sandbox invoke invoke function not found {}", dlerror());

  shared->token.is_child_executing = false;
  shared->token.is_child_loaded = true;

  static constexpr size_t stack_size = 8 * 1024 * 1024;
  void* stack = malloc(stack_size);
  // Enter the run loop, waiting for calls from trusted code.
  // We do this in a new thread so that our stack can be in the shared region.
  // This avoids the need to copy arguments from the stack to the heap.
  runloop_with_stack_pivot(stack, stack_size);

  return 0;
}
