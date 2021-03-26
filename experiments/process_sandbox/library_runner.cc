// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "child_malloc.h"
#include "helpers.h"
#include "host_service_calls.h"
#include "platform/platform.h"
#include "privilege_elevation_upcalls.h"
#include "sandbox.hh"
#include "shared.h"
#include "shared_memory_region.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
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
    asprintf(&str, "/proc/%d/fd/%d", (int)getpid(), fd);
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

sandbox::ProxyPageMap sandbox::ProxyPageMap::p;

using namespace snmalloc;
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
  uintptr_t
  requestHostService(HostServiceCallID id, uintptr_t arg0, uintptr_t arg1 = 0)
  {
    static std::atomic_flag lock;
    FlagLock g(lock);
    HostServiceRequest req{id, arg0, arg1};
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

MemoryProviderProxy* MemoryProviderProxy::make() noexcept
{
  if (unlikely(!done_bootstrapping))
  {
    bootstrap();
  }
  static MemoryProviderProxy singleton;
  return &singleton;
}

namespace sandbox
{
  void ProxyPageMap::set(uintptr_t p, uint8_t x)
  {
    SANDBOX_DEBUG_INVARIANT(
      is_inside_shared_memory(
        reinterpret_cast<void*>(p), snmalloc::OS_PAGE_SIZE),
      "Setting metadata pointer {} in pagemap that is outside of the sandbox "
      "range {}--{}",
      p,
      shared_memory_start,
      shared_memory_end);
    requestHostService(
      ChunkMapSet, reinterpret_cast<uintptr_t>(p), static_cast<uintptr_t>(x));
  }

  uint8_t ProxyPageMap::get(address_t p)
  {
    return GlobalPagemap::pagemap().get(p);
  }

  uint8_t ProxyPageMap::get(void* p)
  {
    return GlobalPagemap::pagemap().get(address_cast(p));
  }

  void ProxyPageMap::set_slab(snmalloc::Superslab* slab)
  {
    set(reinterpret_cast<uintptr_t>(slab), (size_t)CMSuperslab);
  }

  void ProxyPageMap::clear_slab(snmalloc::Superslab* slab)
  {
    set(reinterpret_cast<uintptr_t>(slab), (size_t)CMNotOurs);
  }

  void ProxyPageMap::set_slab(snmalloc::Mediumslab* slab)
  {
    set(reinterpret_cast<uintptr_t>(slab), (size_t)CMMediumslab);
  }

  void ProxyPageMap::clear_slab(snmalloc::Mediumslab* slab)
  {
    set(reinterpret_cast<uintptr_t>(slab), (size_t)CMNotOurs);
  }

  void ProxyPageMap::set_large_size(void* p, size_t size)
  {
    size_t size_bits = bits::next_pow2_bits(size);
    SANDBOX_DEBUG_INVARIANT(
      is_inside_shared_memory(p, size),
      "Setting large size for pointer {} in pagemap that is outside of the "
      "sandbox range {}--{}",
      p,
      shared_memory_start,
      shared_memory_end);
    requestHostService(
      ChunkMapSetRange,
      reinterpret_cast<uintptr_t>(p),
      static_cast<uintptr_t>(size_bits));
  }

  void ProxyPageMap::clear_large_size(void* p, size_t size)
  {
    SANDBOX_DEBUG_INVARIANT(
      is_inside_shared_memory(p, size),
      "Clearing large size for pointer {} in pagemap that is outside of the "
      "sandbox range {}--{}",
      p,
      shared_memory_start,
      shared_memory_end);
    size_t size_bits = bits::next_pow2_bits(size);
    requestHostService(
      ChunkMapClearRange,
      reinterpret_cast<uintptr_t>(p),
      static_cast<uintptr_t>(size_bits));
  }

  void* MemoryProviderProxy::pop_large_stack(size_t large_class)
  {
    return reinterpret_cast<void*>(requestHostService(
      MemoryProviderPopLargeStack, static_cast<uintptr_t>(large_class)));
  }

  void
  MemoryProviderProxy::push_large_stack(Largeslab* slab, size_t large_class)
  {
    requestHostService(
      MemoryProviderPushLargeStack,
      reinterpret_cast<uintptr_t>(slab),
      static_cast<uintptr_t>(large_class));
  }

  void* MemoryProviderProxy::reserve_committed_size(size_t size) noexcept
  {
    size_t size_bits = snmalloc::bits::next_pow2_bits(size);
    size_t large_class = std::max(size_bits, SUPERSLAB_BITS) - SUPERSLAB_BITS;
    return reserve_committed(large_class);
  }
  void* MemoryProviderProxy::reserve_committed(size_t large_class) noexcept
  {
    return reinterpret_cast<void*>(requestHostService(
      MemoryProviderReserve, static_cast<uintptr_t>(large_class)));
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
   * the library's vtable) as an argument.  Exits when the upcall depth changes
   * after executing the helper function.  This provides a nested runloop
   * abstraction similar to OpenStep's modal runloop. Each recursion depth in
   * an upcall has its own runloop that handles recursive invocations from the
   * parent in response to the upcall.
   */
  void runloop(int upcall_depth = 0)
  {
    int new_depth;
    do
    {
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
        printf("Exception!\n");
      }
      new_depth = shared->token.upcall_depth;
      // Wake up the parent if it's expecting a wakeup for this upcall depth.
      // The `upcall` function has a wake but not a wait because it is using
      // the `wait` in this function, we need to ensure that we don't unbalance
      // the wakes and waits.
      if (new_depth == upcall_depth)
      {
        shared->token.is_child_executing = false;
        shared->token.parent.wake();
      }
    } while (new_depth == upcall_depth);
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

    // fprintf(stderr, "Child starting\n");
    // printf(
    //"Child trying to map fd %d at addr %p (0x%zx)\n", SharedMemRegion, addr,
    // length);
    void* ptr = mmap(
      addr,
      length,
      PROT_READ | PROT_WRITE,
      MAP_FIXED_NOREPLACE | MAP_SHARED | platform::detail::map_nocore,
      SharedMemRegion,
      0);

    // printf("%p\n", ptr);
    if (ptr == MAP_FAILED)
    {
      err(1, "Mapping shared heap failed");
    }

    shared = reinterpret_cast<SharedMemoryRegion*>(ptr);
    // Splice the pagemap page inherited from the parent into the pagemap.
    void* pagemap_chunk = GlobalPagemap::pagemap().page_for_address(
      reinterpret_cast<uintptr_t>(ptr));
    munmap(pagemap_chunk, snmalloc::OS_PAGE_SIZE);
    void* shared_pagemap = mmap(
      pagemap_chunk,
      snmalloc::OS_PAGE_SIZE,
      PROT_READ,
      MAP_SHARED | MAP_FIXED,
      PageMapPage,
      0);
    if (shared_pagemap == MAP_FAILED)
    {
      err(1, "Mapping shared pagemap page failed");
    }
    shared_memory_start = shared->start;
    shared_memory_end = shared->end;
    SANDBOX_INVARIANT(
      shared_pagemap == pagemap_chunk,
      "Mapping pagmemap chunk failed.  Expected {}, got {}",
      pagemap_chunk,
      shared_pagemap);
    (void)shared_pagemap;

    done_bootstrapping = true;
  }

  using sandbox::platform::Handle;
  using Socket = sandbox::platform::SocketPair::Socket;

  /**
   * The socket that is used for upcalls to the parent process.
   */
  Socket upcallSocket;

  /**
   * Perform an upcall.  This takes the kind of upcall, the data to be sent,
   * and the file descriptor to send as arguments.  The file descriptor may be
   * -1, in which case the it is not sent.
   *
   *  The return value is the integer result of the upcall and a `Handle` that
   *  is either invalid or the returned file descriptor.
   *
   *  This function should not be called directly, it should be invoked via the
   *  wrapper.
   */
  std::pair<uintptr_t, Handle>
  upcall(sandbox::UpcallKind k, void* buffer, size_t size, int fd)
  {
    Handle out_fd(fd);
    UpcallRequest req{k, size, reinterpret_cast<uintptr_t>(buffer)};
    upcallSocket.send(&req, sizeof(req), out_fd);
    out_fd.take();
    int depth = ++shared->token.upcall_depth;
    shared->token.is_child_executing = false;
    shared->token.parent.wake();
    runloop(depth);
    Handle in_fd;
    UpcallResponse response;
    upcallSocket.receive(&response, sizeof(response), in_fd);
    return {response.response, std::move(in_fd)};
  }

  /**
   * Perform an upcall, of the specified kind, passing `data`.  The `data`
   * argument must point to the shared heap.
   *
   * If the optional `fd` parameter is passed, then this file descriptor
   * accompanies the upcall.  This is used for calls such as `openat`.
   */
  template<typename T>
  std::pair<uintptr_t, Handle>
  upcall(sandbox::UpcallKind k, T* data, int fd = -1)
  {
    return upcall(k, data, sizeof(T), fd);
  }

  /**
   * Emulate the `stat` system call by performing an upcall to the parent.
   */
  int upcall_stat(const char* pathname, struct stat* statbuf)
  {
    auto args = std::make_unique<sandbox::UpcallArgs::Stat>();
    unique_c_ptr<char> copy;
    if (!is_inside_shared_memory(pathname))
    {
      copy.reset(strdup(pathname));
      pathname = copy.get();
    }
    args->path = reinterpret_cast<uintptr_t>(pathname);
    args->statbuf = reinterpret_cast<uintptr_t>(statbuf);
    auto ret = upcall(sandbox::UpcallKind::Stat, args.get());
    return static_cast<int>(ret.first);
  }

  /**
   * Emulate the `open` system call by performing an upcall to the parent.
   */
  int upcall_open(const char* pathname, int flags, mode_t mode)
  {
    auto args = std::make_unique<sandbox::UpcallArgs::Open>();
    unique_c_ptr<char> copy;
    if (!is_inside_shared_memory(pathname))
    {
      copy.reset(strdup(pathname));
      pathname = copy.get();
    }
    args->path = reinterpret_cast<uintptr_t>(pathname);
    args->flags = flags;
    args->mode = mode;
    auto ret = upcall(sandbox::UpcallKind::Open, args.get());
    int result = static_cast<int>(ret.first);
    if (ret.second.is_valid())
    {
      result = ret.second.take();
    }
    return result;
  }

  /**
   * The upcall functions use the Linux convention for system call returns:
   * non-negative numbers indicate success, negative numbers indicate valuesw
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
   * Emulate the `openat` system call by performing an upcall to the parent.
   */
  int upcall_openat(int dirfd, const char* pathname, int flags, mode_t mode)
  {
    (void)dirfd;
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
      return upcall_open(pathname, flags, mode);
    }
    // TODO: Perform an upcall for the openat emulation.
    return -EINVAL;
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
    get<Arg>(args) = c.get_arg<
      Arg,
      std::tuple_element_t<Arg, std::remove_reference_t<Tuple>>>();
    if constexpr (Arg > 0)
    {
      extract_args<Tuple, Arg - 1>(args, c);
    }
  }

  /**
   * Signal handler function.  For system calls that are emulated after a trap,
   * this extracts the arguments from the trap frame, calls the correct upcall
   * function, and then injects the return address into the syscall frame.
   */
  void emulate(int, siginfo_t* info, ucontext_t* ctx)
  {
    SyscallFrame c(*info, *ctx);
    if (c.is_sandbox_policy_violation())
    {
      int syscall = c.get_syscall_number();
      auto call = [&](auto&& fn) {
        typename sandbox::internal::signature<decltype(fn)>::argument_type args;
        extract_args(args, c);
        return std::apply(fn, args);
      };
      auto syscall_upcall = [&](int number, auto&& fn) {
        if ((number != -1) && (number == syscall))
        {
          uintptr_t result = call(fn);
          if (result < 0)
          {
            c.set_error_return(-result);
          }
          else
          {
            c.set_success_return(result);
          }
          return;
        }
      };
      syscall_upcall(SyscallFrame::Open, upcall_open);
      syscall_upcall(SyscallFrame::OpenAt, upcall_openat);
      syscall_upcall(SyscallFrame::Stat, upcall_stat);
    }
  }

}

/**
 * POSIX `open` function, performs an upcall to the host rather than a system
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
  return syscall_return(upcall_open(pathname, flags, mode));
}

#ifndef USE_CAPSICUM
/**
 * POSIX `openat` function, performs an upcall to the host rather than a system
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
  int ret = upcall_openat(dirfd, pathname, flags, mode);
  if (ret < 0)
  {
    errno = -ret;
    return -1;
  }
  return ret;
}
#endif

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
  auto ret = upcall(static_cast<sandbox::UpcallKind>(idx), data, size, fd);
  int result = static_cast<int>(ret.first);
  if (ret.second.is_valid())
  {
    result = ret.second.take();
  }
  return result;
}

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
  upcallSocket.reset(FDSocket);

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
  check_is_in_shared_range(current_alloc_pool());
  check_is_in_shared_range(ThreadAlloc::get_reference());
  void* obj = malloc(42);
  check_is_in_shared_range(obj);
  free(obj);
  fprintf(stderr, "Sandbox: %p--%p\n", shared_memory_start, shared_memory_end);
#endif

  // Load the library using the file descriptor that the parent opened.  This
  // allows a Capsicum sandbox to prevent any access to the global namespace.
  // It is hopefully possible to implement something similar with seccomp-bpf,
  // though this may require calling into the parent to request additional file
  // descriptors and proxying all open / openat calls.
  void* handle = fdlopen(MainLibrary, RTLD_GLOBAL | RTLD_LAZY);
  if (handle == nullptr)
  {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
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

  // Enter the run loop, waiting for calls from trusted code.
  runloop();

  return 0;
}
