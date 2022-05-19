// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <array>
#include <chrono>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef __unix__
#  include <dlfcn.h>
#  include <err.h>
#  include <fcntl.h>
#  include <libgen.h>
#  include <stdio.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/un.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  ifdef __linux__
#    include <bsd/unistd.h>
#  endif
#endif
#include "host_service_calls.h"
#include "process_sandbox/callbacks.h"
#include "process_sandbox/filetree.h"
#include "process_sandbox/netpolicy.h"
#include "process_sandbox/path.h"
#include "process_sandbox/platform/sandbox.h"
#include "process_sandbox/sandbox.h"
#include "process_sandbox/shared_memory_region.h"

// Linux doesn't provide a useful error code for an invalid hints pointer, use
// the closest approximation.
#ifndef EAI_BADHINTS
#  define EAI_BADHINTS EAI_BADFLAGS
#endif

using namespace std::chrono_literals;

namespace
{
  // The library load paths.  We're going to pass all of these to the
  // child as open directory descriptors for the run-time linker to use.
  std::array<const char*, 3> libdirs = {"/lib", "/usr/lib", "/usr/local/lib"};
}

namespace sandbox
{
  SharedAllocConfig::LocalState::LocalState(void* start, size_t size)
  : base(start), top(pointer_offset(base, size))
  {
    // Force initialisation of the shared memory object backing the pagemap.
    SharedAllocConfig::ensure_initialised();
    snmalloc::range_to_pow_2_blocks<snmalloc::MIN_CHUNK_BITS>(
      snmalloc::capptr::Chunk<void>(start),
      size,
      [&](snmalloc::capptr::Chunk<void> p, size_t sz, bool) {
        auto [g, m] = get_memory();
        m.dealloc_range(p, sz);
      });
  }
  /**
   * Singleton class that handles pagemap updates from children.  This listens
   * on a socket for updates, validates that they correspond to the memory that
   * this child is responsible for, and if so updates both that child's shared
   * pagemap page and the parent process's pagemap.
   *
   * This class creates a new thread in the background that waits for pagemap
   * updates and processes them.
   */
  class MemoryServiceProvider
  {
    platform::Poller poller;
    /**
     * Add a new socket that we'll wait for.  This can be called from any
     * thread without synchronisation.
     */
    void register_fd(platform::Handle& socket)
    {
      poller.add(socket.fd);
    }
    /**
     * Mutex that protects the `ranges` map.
     */
    std::mutex m;

    /**
     * A map from file descriptor over which we've received an update request
     * to the sandbox metadata.
     *
     * FIXME: In C++20, we can perform a lookup by `handle_t` even if the key
     * value is a `Handle`, but we can't in earlier C++ so instead we fudge it
     * by storing an owning copy of the handle in the value and using the
     * non-owning value in the value.
     */
    std::unordered_map<
      platform::handle_t,
      std::pair<platform::SocketPair::Socket, Library*>>
      ranges;
    /**
     * Run loop.  Wait for updates from the child.
     */
    void run()
    {
      int fd;
      bool eof;
      while (poller.poll(fd, eof))
      {
        auto remove_fd = [&]() {
          std::lock_guard g(m);
          auto r = ranges.find(fd);
          if (r != ranges.end())
          {
            ranges.erase(r);
          }
        };
        // If a child's socket closed, unmap its shared page and delete the
        // metadata that we have associated with it.
        if (eof)
        {
          remove_fd();
          continue;
        }
        decltype(ranges)::mapped_type* entry;
        {
          decltype(ranges)::iterator r;
          std::lock_guard g(m);
          r = ranges.find(fd);
          if (r == ranges.end())
          {
            continue;
          }
          entry = &r->second;
        }
        auto& sock = entry->first;
        Library* lib = entry->second;
        SharedAllocConfig::LocalState* s = &lib->memory_provider;
        HostServiceRequest rpc;
        if (!sock.nonblocking_receive(rpc))
        {
          remove_fd();
          lib->terminate();
          continue;
        }
        HostServiceResponse reply{0, 0};
        auto is_metaentry_valid =
          [&](size_t size, SharedAllocConfig::Pagemap::Entry& metaentry) {
            auto sizeclass = metaentry.get_sizeclass();
            auto remote = metaentry.get_remote();
            return ((remote == nullptr) ||
                    s->contains(remote, sizeof(snmalloc::RemoteAllocator))) &&
              (snmalloc::sizeclass_full_to_size(sizeclass) <= size);
          };
        // No default so we get range checking.  Fallthrough returns the error
        // result.
        switch (rpc.kind)
        {
          case AllocChunk:
          {
            auto size = static_cast<size_t>(rpc.args[0]);
            if (
              (size < snmalloc::MIN_CHUNK_SIZE) ||
              !snmalloc::bits::is_pow2(size))
            {
              reply.error = 3;
              break;
            }
            auto meta =
              reinterpret_cast<SharedAllocConfig::SlabMetadata*>(rpc.args[1]);
            auto ras = rpc.args[2];
            // `meta` refers to the pointer to the slab metadata.  This field in
            // the `Entry` is dereferenced outside of the sandbox only in the
            // case where the remote is not the single remote of the allocator
            // associated with this sandbox for use on the outside.
            SharedAllocConfig::Pagemap::Entry metaentry{meta, ras};
            if (!is_metaentry_valid(size, metaentry))
            {
              reply.error = 1;
              break;
            }
            snmalloc::capptr::Chunk<void> alloc;
            {
              auto [g, m] = s->get_memory();
              alloc = m.alloc_range(size);
            }
            if (alloc == nullptr)
            {
              reply.error = 2;
              break;
            }
            metaentry.claim_for_sandbox();
            SharedAllocConfig::Pagemap::set_metaentry(
              address_cast(alloc), size, metaentry);

            reply.ret = alloc.unsafe_uintptr();
            break;
          }
          case DeallocChunk:
          {
            snmalloc::capptr::Chunk<void> ptr{
              reinterpret_cast<void*>(rpc.args[0])};
            size_t size = static_cast<size_t>(rpc.args[1]);
            if (!s->contains(ptr.unsafe_ptr(), size))
            {
              reply.error = 1;
              break;
            }
            // The size must be a power of two, larger than the chunk size
            if (!(snmalloc::bits::is_pow2(size) &&
                  (size >= snmalloc::MIN_CHUNK_SIZE)))
            {
              reply.error = 2;
              break;
            }
            // The base must be chunk-aligned
            if (
              snmalloc::pointer_align_down(
                ptr.unsafe_ptr(), snmalloc::MIN_CHUNK_SIZE) != ptr.unsafe_ptr())
            {
              reply.error = 3;
              break;
            }
            auto address = snmalloc::address_cast(ptr);
            for (size_t chunk_offset = 0; chunk_offset < size;
                 chunk_offset += snmalloc::MIN_CHUNK_SIZE)
            {
              auto& meta = SharedAllocConfig::Pagemap::get_metaentry_mut(
                address + chunk_offset);
              if (!meta.is_sandbox_owned())
              {
                reply.error = 4;
                break;
              }
            }
            if (reply.error == 0)
            {
              SharedAllocConfig::dealloc_range(*s, ptr, size);
            }
            break;
          }
        }
        // If we can't do a non-blocking send, then the child must have filled
        // up the kernel's buffer, kill the child.
        if (!sock.nonblocking_send(reply))
        {
          remove_fd();
          lib->terminate();
        }
      }
      err(1, "Waiting for pagetable updates failed");
    }

  public:
    /**
     * Constructor.  Spawns a background thread to run and process updates.
     */
    MemoryServiceProvider()
    {
      std::thread t([&]() { run(); });
      t.detach();
    }
    /**
     * Notify this class that a sandbox exists.  The `start` and `end`
     * parameters indicate the address range assigned to this sandbox.
     * `socket_fd` provides the file descriptor for the socket over which the
     * sandbox will send update requests.  `pagemap_fd` is the shared pagemap
     * page.
     */
    void add_range(Library& sandbox, platform::SocketPair::Socket&& socket)
    {
      {
        std::lock_guard g(m);
        register_fd(socket);
        platform::handle_t socket_fd = socket.fd;
        ranges.emplace(socket_fd, std::make_pair(std::move(socket), &sandbox));
      }
    }
  };
  /**
   * Return a singleton instance of the pagemap owner.
   */
  MemoryServiceProvider& memory_service_provider()
  {
    // Leaks.  No need to run the destructor!
    static MemoryServiceProvider* p = new MemoryServiceProvider();
    return *p;
  }

  /**
   * Class that handles callbacks.  Each `Library` holds a single one
   * of these, its implementation is hidden from the public interface.
   */
  class CallbackDispatcher
  {
    /**
     * The file paths exported to this sandbox.
     */
    ExportedFileTree vfs;

    /**
     * The policy for network access.
     */
    NetworkPolicy netpolicy;

    /**
     * Vector of callback handlers.
     */
    std::vector<std::unique_ptr<CallbackHandlerBase>> handlers;

    /**
     * This is an implementation detail of `Library`,
     * `Library` may call any of it.
     */
    friend class Library;

    /**
     * The handle to the socket that is used to pass file descriptors to the
     * sandboxed process.
     */
    platform::SocketPair::Socket socket;

    /**
     * Import the type used for callback returns.
     */
    using Result = CallbackHandlerBase::Result;

    /**
     * Convert a raw number from a system call return into either a file
     * descriptor or the error value.
     */
    Result return_fd(int fd)
    {
      if (fd >= 0)
      {
        platform::Handle h(fd);
        return h;
      }
      return -errno;
    };

    /**
     * Helper for returning an integer value that comes from a system call
     * return.  If the system call fails, returns negated `errno`.
     */
    Result return_int(int ret)
    {
      return ret >= 0 ? ret : -errno;
    };

    /**
     * Copy the path out of the sandbox.
     */
    unique_c_ptr<char> get_path(Library& lib, uintptr_t inSandboxPath)
    {
      return lib.strdup_out(reinterpret_cast<char*>(inSandboxPath));
    };

    /**
     * Check a pointer.  Returns `nullptr` if an object of type `T` at the
     * given address is not fully contained within the sandbox.  Returns a
     * pointer to the object cast to the correct type.
     *
     * This does *not* copy and so code should not read from any byte in the
     * argument more than once or it will be subject to TOCTOU errors.
     */
    template<typename T>
    T* check_pointer(Library& lib, uintptr_t addr)
    {
      T* ptr = reinterpret_cast<T*>(addr);
      if (lib.contains(ptr, sizeof(T)))
      {
        return ptr;
      }
      return nullptr;
    }

    template<typename T>
    Result
    handle_path_syscall(Library& lib, uintptr_t inSandboxPath, T&& handler)
    {
      auto raw_path = get_path(lib, inSandboxPath);
      if (!raw_path)
      {
        return {-EINVAL};
      }
      Path path{raw_path.get()};
      if (!path.canonicalise())
      {
        return {-EINVAL};
      }
      auto allowed = vfs.lookup_file(path);
      if (!allowed.has_value())
      {
        return {-ENOENT};
      }
      auto [fd, path_tail] = allowed.value();
      return handler(fd, path_tail);
    }

    /**
     * Handle an `open` callback by reading the file from the exported file
     * tree.
     */
    Result handle_open(Library& lib, SyscallArgs<Open>::rpc_type& args)
    {
      return handle_path_syscall(
        lib, std::get<0>(args), [&](auto fd, auto& path_tail) {
          if (!path_tail.is_empty())
          {
            fd = platform::SafeSyscalls::openat_beneath(
              fd,
              path_tail.str().c_str(),
              std::get<1>(args),
              std::get<2>(args));
          }
          else
          {
            // FIXME: Ugly hack to work around the lack of a non-owning version
            // of `Handle`
            fd = ::dup(fd);
          }
          return return_fd(fd);
        });
    }

    /**
     * Handle an `openat` callback by reading the file from the exported file
     * tree.
     */
    Result handle_openat(
      Library& lib, SyscallArgs<OpenAt>::rpc_type& args, platform::Handle h)
    {
      return platform::SafeSyscalls::openat_beneath(
        h.fd,
        get_path(lib, std::get<1>(args)).get(),
        std::get<2>(args),
        std::get<3>(args));
    }

    /**
     * Handle either a bind or connect call.  These take the same set of
     * arguments and so can be handled with common code before dispatching to
     * the correct policy function.
     */
    template<CallbackKind K, NetworkPolicy::NetOperation Op>
    Result handle_bind_or_connect(
      Library& lib, typename SyscallArgs<K>::rpc_type& args, platform::Handle h)
    {
      // Don't allow an attacker to force us to copy huge things.  The size
      // of a sockaddr is on the order of a few tens of bytes, clamp this to
      // a size that is well over the biggest that we expect.
      constexpr size_t maxSaneSockAddrSize = 4096;
      const void* unsafeBase = reinterpret_cast<void*>(std::get<1>(args));
      const socklen_t length = std::get<2>(args);
      if (
        !h.is_valid() || (length > maxSaneSockAddrSize) ||
        !lib.contains(unsafeBase, length))
      {
        return {-EINVAL};
      }
      char buffer[maxSaneSockAddrSize];
      memcpy(buffer, unsafeBase, length);
      int ret =
        netpolicy.invoke<Op>(h.fd, reinterpret_cast<sockaddr*>(buffer), length);
      return return_int(ret);
    }

    /**
     * Handle the `getaddrinfo` libc call.  Note that this is not a system call
     * and so does not follow the same rules for integer return values as system
     * call.
     */
    Result
    handle_getaddrinfo(Library& lib, SyscallArgs<GetAddrInfo>::rpc_type& args)
    {
      // If the result pointer is not valid, report an error
      addrinfo** unsafeSandboxRes =
        reinterpret_cast<addrinfo**>(std::get<3>(args));
      if (!lib.contains(unsafeSandboxRes, sizeof(addrinfo*)))
      {
        return EAI_MEMORY;
      }
      addrinfo* unsafeHints = reinterpret_cast<addrinfo*>(std::get<2>(args));
      addrinfo hintsCopy;
      addrinfo* hints;
      if (unsafeHints == nullptr)
      {
        hints = nullptr;
      }
      else
      {
        if (!lib.contains(unsafeHints, sizeof(addrinfo)))
        {
          return EAI_BADHINTS;
        }
        // The `hints` argument to `getaddrinfo` may contain the `ai_flags`,
        // `ai_family`, `ai_socktype`, and `ai_protocol` fields.  These are not
        // pointers and so can be shallow copied.  Some of the other fields
        // *are* pointers, so explicitly zero them here.  If bugs in any of the
        // code that receives the copy of `hints` reads the fields that it
        // should not, then this prevents the untrusted code from being able to
        // influence the pointers that buggy code on the trusted side copies.
        memset(&hintsCopy, 0, sizeof(hintsCopy));
        hintsCopy.ai_flags = unsafeHints->ai_flags;
        hintsCopy.ai_family = unsafeHints->ai_family;
        hintsCopy.ai_socktype = unsafeHints->ai_socktype;
        hintsCopy.ai_protocol = unsafeHints->ai_protocol;
        hints = &hintsCopy;
      }
      auto host = get_path(lib, std::get<0>(args));
      auto service = get_path(lib, std::get<1>(args));
      addrinfo* res;
      int ret = netpolicy.invoke<NetworkPolicy::NetOperation::GetAddrInfo>(
        host.get(), service.get(), hints, &res);
      // On success, we need to copy the result out and free it.
      if (ret == 0)
      {
        size_t count = 0;
        size_t extra = 0;
        // Find the number of `addrinfo` structures in the list and the size of
        // their associated `sockaddr`s
        for (addrinfo* cur = res; cur != nullptr; cur = cur->ai_next)
        {
          count++;
          extra += cur->ai_addrlen;
        }
        if (count > 0)
        {
          // Allocate space for everything.  No overflow checking here, but the
          // called code is trusted and we could only overflow if it returned a
          // nonsense `ai_addrlen`.
          char* buffer = lib.alloc<char>((sizeof(addrinfo) * count) + extra);
          // The allocated space will be an array of `addrinfo`s, followed by
          // (variable-sized) `sockaddr`s.
          addrinfo* ais = reinterpret_cast<addrinfo*>(buffer);
          char* sockaddrs = buffer + (sizeof(addrinfo) * count);
          size_t i = 0;
          // Copy each list element into the new buffer, updating its next
          // pointer and copying the payload into the end of the buffer.
          for (addrinfo* cur = res; cur != nullptr; cur = cur->ai_next, i++)
          {
            ais[i] = *cur;
            memcpy(sockaddrs, cur->ai_addr, cur->ai_addrlen);
            ais[i].ai_addr = reinterpret_cast<sockaddr*>(sockaddrs);
            sockaddrs += cur->ai_addrlen;
            ais[i].ai_next = &ais[i + 1];
          }
          // Add the null terminator in the list.
          ais[count - 1].ai_next = nullptr;
          *unsafeSandboxRes = ais;
          netpolicy.freeaddrinfo(res);
        }
      }
      return return_int(ret);
    }

    /**
     * Handle an `access` callback by checking the file in the exported file
     * tree.
     */
    Result handle_access(Library& lib, SyscallArgs<Access>::rpc_type& args)
    {
      return handle_path_syscall(
        lib, std::get<0>(args), [&](auto fd, auto& path_tail) {
          return return_int(platform::SafeSyscalls::faccessat_beneath(
            fd,
            path_tail.is_empty() ? nullptr : path_tail.str().c_str(),
            std::get<1>(args)));
        });
    }

    /**
     * Handle a `stat` callback by forwarding to a real `fstat` call if the
     * exported file tree provides a file descriptor corresponding to this
     * path.
     */
    Result handle_stat(Library& lib, SyscallArgs<Stat>::rpc_type& args)
    {
      return handle_path_syscall(
        lib, std::get<0>(args), [&](auto fd, auto& path_tail) {
          struct stat* sb = check_pointer<struct stat>(lib, std::get<1>(args));
          if (sb != nullptr)
          {
            return return_int(
              path_tail.is_empty() ? fstat(fd, sb) :
                                     platform::SafeSyscalls::fstatat_beneath(
                                       fd, path_tail.str().c_str(), sb, 0));
          }
          return Result{-EINVAL};
        });
    }

    /**
     * Helper, enlarges the handlers array, filling it in with empty handlers.
     */
    void enlarge_handlers(size_t size)
    {
      if (size >= handlers.size())
      {
        size_t oldsize = handlers.size();
        handlers.resize(size);
        for (size_t i = oldsize; i < size; i++)
        {
          handlers[i] = std::make_unique<CallbackHandlerBase>();
        }
      }
    }

    /**
     * Register a handler for an callback, with the specific index.  Note that,
     * although `k` is an `CallbackKind`, this will be a value after the last
     * statically defined callback kind for any user-provided callbacks.
     */
    template<typename Args>
    void register_handler(
      CallbackKind k, Result (CallbackDispatcher::*handler)(Library&, Args&))
    {
      enlarge_handlers(k + 1);
      handlers[k] = make_callback_handler<Args>(
        std::bind(handler, this, std::placeholders::_1, std::placeholders::_2));
    }

    /**
     * Register a handler for an callback, with the specific index.  Note that,
     * although `k` is an `CallbackKind`, this will be a value after the last
     * statically defined callback kind for any user-provided callbacks.
     */
    template<typename Args>
    void register_handler(
      CallbackKind k,
      Result (CallbackDispatcher::*handler)(Library&, Args&, platform::Handle))
    {
      enlarge_handlers(k + 1);
      handlers[k] = make_callback_handler<Args, true>(std::bind(
        handler,
        this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3));
    }

    /**
     * Handle a request.
     */
    void handle(Library& lib)
    {
      CallbackRequest req;
      platform::Handle in_fd;
      // This should not block, but it can if the sandbox doesn't write
      // anything into the socket.
      if (!socket.nonblocking_receive(req, in_fd))
      {
        return;
      }

      CallbackHandlerBase::Result ret;
      if (req.kind < handlers.size())
      {
        ret = handlers[req.kind]->invoke(lib, req, std::move(in_fd));
      }
      if (!socket.nonblocking_send(ret.integer, ret.handle))
      {
        lib.terminate();
      }
    }

    /**
     * The next callback number to use.
     */
    int next_callback_number = CallbackKind::FirstUserFunction;

    /**
     * Register an callback for the next available user-defined callback number.
     */
    int register_callback(std::unique_ptr<CallbackHandlerBase>&& callback)
    {
      int n = next_callback_number++;
      enlarge_handlers(n + 1);
      handlers[n] = std::move(callback);
      return n;
    }

  public:
    /**
     * Constructor.  Set up the default exported directories.
     */
    CallbackDispatcher()
    {
      for (auto libdir : libdirs)
      {
        int fd = open(libdir, O_DIRECTORY);
        if (fd > 0)
        {
          vfs.add_directory(libdir, platform::Handle(fd));
        }
      }
      const char* ldsocache = "/etc/ld.so.cache";
      int fd = open(ldsocache, O_RDONLY);
      if (fd > 0)
      {
        vfs.add_file(ldsocache, platform::Handle(fd));
      }
      handlers.reserve(CallbackKind::BuiltInCallbackKindCount);
      register_handler(CallbackKind::Open, &CallbackDispatcher::handle_open);
      register_handler(CallbackKind::Stat, &CallbackDispatcher::handle_stat);
      register_handler(
        CallbackKind::Access, &CallbackDispatcher::handle_access);
      register_handler(
        CallbackKind::OpenAt, &CallbackDispatcher::handle_openat);
      register_handler(
        CallbackKind::Bind,
        &CallbackDispatcher::handle_bind_or_connect<
          CallbackKind::Bind,
          NetworkPolicy::NetOperation::Bind>);
      register_handler(
        CallbackKind::GetAddrInfo, &CallbackDispatcher::handle_getaddrinfo);
      register_handler(
        CallbackKind::Connect,
        &CallbackDispatcher::handle_bind_or_connect<
          CallbackKind::Connect,
          NetworkPolicy::NetOperation::Connect>);
    };
  };

  ExportedFileTree& Library::filetree()
  {
    return callback_dispatcher->vfs;
  }

  NetworkPolicy& Library::network_policy()
  {
    return callback_dispatcher->netpolicy;
  }

  int Library::register_callback(
    std::unique_ptr<CallbackHandlerBase>&& callback)
  {
    return callback_dispatcher->register_callback(std::move(callback));
  }

  Library::~Library()
  {
    wait_for_child_exit();
    {
      auto [g, pm] = SharedAllocConfig::Pagemap::get_pagemap_writeable();
      snmalloc::address_t base =
        snmalloc::address_cast(memory_provider.base_address());
      auto top = snmalloc::address_cast(memory_provider.top_address());
      SharedAllocConfig::Pagemap::Entry empty{nullptr, 0};
      // Scan the pagemap for all memory associated with this and deallocate
      // the metaslabs.  Note that we don't need to do any cleanup for the
      // memory referenced by these metaslabs: it will all go away when the
      // shared memory region is deallocated.
      for (snmalloc::address_t a = base; a < top; a += snmalloc::MIN_CHUNK_SIZE)
      {
        auto& meta = SharedAllocConfig::Pagemap::get_metaentry_mut(a);
        if (!meta.is_backend_owned())
        {
          auto* remote = meta.get_remote();
          if (!meta.is_sandbox_owned() && (remote != nullptr))
          {
            delete meta.get_slab_metadata();
          }
        }
        meta = empty;
        SANDBOX_DEBUG_INVARIANT(
          !meta.is_sandbox_owned(),
          "Unused pagemap entry must not be sandbox owned");
      }
    }
    shared_mem->destroy();
  }

  void Library::start_child(
    const char* library_name,
    const char* librunnerpath,
    const void* sharedmem_addr,
    const platform::Handle& pagemap_mem,
    platform::Handle&& malloc_rpc_socket,
    platform::Handle&& fd_socket)
  {
    static const int last_fd = OtherLibraries;
    auto move_fd = [](int x) {
      SANDBOX_DEBUG_INVARIANT(
        x >= 0, "Attempting to move invalid file descriptor {}", x);
      while (x < last_fd)
      {
        x = dup(x);
      }
      return x;
    };
    // Move all of the file descriptors that we're going to use out of the
    // region that we're going to populate.
    int shm_fd = move_fd(shm.get_handle().take());
    int pagemap_fd = move_fd(pagemap_mem.fd);
    fd_socket = move_fd(fd_socket.take());
    malloc_rpc_socket = move_fd(malloc_rpc_socket.take());
    // Open the library binary.  If this fails, kill the child process.  Note
    // that we do this *before* dropping privilege - we don't have to give
    // the child the right to look in the directory that contains this
    // binary.
    int library = open(library_name, O_RDONLY);
    if (library < 0)
    {
      _exit(-1);
    }
    library = move_fd(library);
    // The child process expects to find these in fixed locations.
    shm_fd = dup2(shm_fd, SharedMemRegion);
    dup2(pagemap_fd, PageMapPage);
    dup2(fd_socket.take(), FDSocket);
    assert(library);
    library = dup2(library, MainLibrary);
    assert(library == MainLibrary);
    dup2(malloc_rpc_socket.take(), PageMapUpdates);
    closefrom(last_fd);
    // Prepare the arguments to main.  These are going to be the binary name,
    // the address of the shared memory region, the length of the shared
    // memory region, and a null terminator.  We have to pass the two
    // addresses as strings because the kernel will assume that all arguments
    // to main are null-terminated strings and will copy them into the
    // process initialisation structure.
    // Note that we create these strings on the stack, rather than calling
    // asprintf, because (if we used vfork) we're still in the same address
    // space as the parent, so if we allocate memory here then it will leak in
    // the parent.
    char location[52];
    size_t loc_len = snprintf(
      location,
      sizeof(location),
      "SANDBOX_LOCATION=%zx:%zx",
      (size_t)sharedmem_addr,
      (size_t)shm.get_size());
    SANDBOX_INVARIANT(
      loc_len < sizeof(location),
      "Location length {} is smaller than expected {}",
      loc_len,
      sizeof(location));
    static_assert(
      OtherLibraries == 8, "First entry in LD_LIBRARY_PATH_FDS is incorrect");
    std::array<const char*, 2> env = {location, nullptr};
    platform::disable_aslr();
    platform::Sandbox::execve(librunnerpath, env, libdirs);
    // Should be unreachable, but just in case we failed to exec, don't return
    // from here (returning from a vfork context is very bad!).
    _exit(EXIT_FAILURE);
  }

  Library::Library(const char* library_name, size_t size)
  : shm(snmalloc::bits::next_pow2_bits(size << 30)),
    memory_provider(
      pointer_offset(shm.get_base(), sizeof(SharedMemoryRegion)),
      shm.get_size() - sizeof(SharedMemoryRegion)),
    callback_dispatcher(std::make_unique<CallbackDispatcher>())
  {
    void* shm_base = shm.get_base();
    // Allocate the shared memory region and set its memory provider to use all
    // of the space after the end of the header for subsequent allocations.
    shared_mem = new (shm_base) SharedMemoryRegion();
    shared_mem->start = shm_base;
    shared_mem->end = pointer_offset(shm.get_base(), shm.get_size());

    // Create a pair of sockets that we can use to
    auto malloc_rpc_sockets = platform::SocketPair::create();
    memory_service_provider().add_range(
      *this, std::move(malloc_rpc_sockets.first));
    // Construct a UNIX domain socket.  This is used to send file descriptors
    // from the parent to the child
    auto socks = platform::SocketPair::create();
    std::string path = ".";
    std::string lib;
    // Use dladdr to find the path of the libsandbox shared library.  For now,
    // we assume that the library runner is in the same place and so is the
    // library that we're going to open.  Eventually we should look for
    // library_runner somewhere else (e.g. ../libexec) and search
    // LD_LIBRARY_PATH for the library that we're going to open.
    Dl_info info;
    static char x;
    if (dladdr(&x, &info))
    {
      char* libpath = ::strdup(info.dli_fname);
      path = dirname(libpath);
      ::free(libpath);
    }
    if (library_name[0] == '/')
    {
      lib = library_name;
    }
    else
    {
      lib = path;
      lib += '/';
      lib += library_name;
    }
    library_name = lib.c_str();
    path += "/library_runner";
    const char* librunnerpath = path.c_str();
    child_proc = std::make_unique<platform::ChildProcess>([&]() {
      // In the child process.
      start_child(
        library_name,
        librunnerpath,
        shm_base,
        SharedAllocConfig::Pagemap::get_pagemap_handle(),
        std::move(malloc_rpc_sockets.second),
        std::move(socks.second));
    });
    callback_dispatcher->socket = std::move(socks.first);
    // Allocate an allocator in the shared memory region.

    allocator = std::make_unique<SharedAlloc>();
    core_alloc = std::make_unique<snmalloc::CoreAllocator<SharedAllocConfig>>(
      &allocator->get_local_cache(), &memory_provider);
    core_alloc->init_message_queue(&shared_mem->allocator_state);
    allocator->init(core_alloc.get());
  }

  void Library::send(int idx, void* ptr)
  {
    // If this is the first call, we need to handle callbacks while the sandbox
    // initialises
    if (is_first_call)
    {
      while (!shared_mem->token.is_child_loaded)
      {
        if (shared_mem->token.callback_depth > 0)
        {
          shared_mem->token.parent.wait(INT_MAX);
          callback_dispatcher->handle(*this);
          shared_mem->token.callback_depth--;
          shared_mem->token.is_child_executing = true;
          shared_mem->token.child.wake();
        }
        else
        {
          std::this_thread::sleep_for(1ms);
        }
      }
    }
    int callback_depth = shared_mem->token.callback_depth.load();
    shared_mem->function_index = idx;
    shared_mem->msg_buffer = ptr;
    assert(!shared_mem->token.is_child_executing);
    shared_mem->token.is_child_executing = true;
    shared_mem->token.child.wake();
    bool handled_callback;
    // Wait for a second, see if the child has exited, if it's still going,
    // try again.
    // FIXME: We should probably allow the user to specify a maxmimum
    // execution time for all calls and kill the sandbox and raise an
    // exception if it's taking too long.
    do
    {
      handled_callback = false;
      while (!shared_mem->token.parent.wait(100))
      {
        if (has_child_exited())
        {
          throw std::runtime_error("Sandboxed library terminated abnormally");
        }
      }
      // If we were woken up for an callback, then handle it, wake up the
      // child, and then continue waiting.
      // Note that we may be called recursively by the callback handler to
      // re-invoke something in the child.  That should only happen for user
      // callbacks
      if (shared_mem->token.callback_depth.load() > callback_depth)
      {
        callback_dispatcher->handle(*this);
        shared_mem->token.callback_depth--;
        shared_mem->token.is_child_executing = true;
        shared_mem->token.child.wake();
        handled_callback = true;
      }
    } while (handled_callback);
  }
  bool Library::has_child_exited()
  {
    return child_proc->exit_status().has_exited;
  }

  void Library::terminate()
  {
    if (!has_child_exited())
    {
      child_proc->terminate();
    }
  }

  int Library::wait_for_child_exit()
  {
    auto exit_status = child_proc->exit_status();
    if (exit_status.has_exited)
    {
      return exit_status.exit_code;
    }
    shared_mem->should_exit = true;
    assert(!shared_mem->token.is_child_executing);
    shared_mem->token.is_child_executing = true;
    shared_mem->token.child.wake();
    return child_proc->wait_for_exit().exit_code;
  }

  void* Library::alloc_in_sandbox(size_t bytes, size_t count)
  {
    bool overflow = false;
    size_t sz = snmalloc::bits::umul(bytes, count, overflow);
    if (overflow)
    {
      return nullptr;
    }
    return allocator->alloc(sz);
  }
  void Library::dealloc_in_sandbox(void* ptr)
  {
    allocator->dealloc(ptr);
  }

}
