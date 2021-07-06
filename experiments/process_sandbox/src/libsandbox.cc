// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <array>
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
#include "process_sandbox/path.h"
#include "process_sandbox/platform/sandbox.h"
#include "process_sandbox/sandbox.h"
#include "process_sandbox/shared_memory_region.h"

namespace
{
  // The library load paths.  We're going to pass all of these to the
  // child as open directory descriptors for the run-time linker to use.
  std::array<const char*, 3> libdirs = {"/lib", "/usr/lib", "/usr/local/lib"};
}

namespace sandbox
{
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
    snmalloc::DefaultChunkMap<> pm;
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
     * Metadata about a sandbox for which we are updating the page map.
     */
    struct Sandbox
    {
      /**
       * The memory provider that owns the above range.
       */
      SharedMemoryProvider* memory_provider;
      /**
       * The shared pagemap page that we need to update on behalf of this
       * process.
       */
      uint8_t* shared_page = nullptr;
    };
    /**
     * A map from file descriptor over which we've received an update request
     * to the sandbox metadata.
     *
     * FIXME: In C++20, we can perform a lookup by `handle_t` even if the key
     * value is a `Handle`, but we can't in earlier C++ so instead we fudge it
     * by storing an owning copy of the handle in the value and using the
     * non-owning value in the value.
     */
    std::unordered_map<platform::handle_t, std::pair<platform::Handle, Sandbox>>
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
        // If a child's socket closed, unmap its shared page and delete the
        // metadata that we have associated with it.
        if (eof)
        {
          std::lock_guard g(m);
          auto r = ranges.find(fd);
          ranges.erase(r);
          continue;
        }
        HostServiceRequest rpc;
        if (
          read(fd, static_cast<void*>(&rpc), sizeof(HostServiceRequest)) !=
          sizeof(HostServiceRequest))
        {
          err(1, "Read from host service pipe %d failed", fd);
          // FIXME: We should kill the sandbox at this point.  It is doing
          // something bad.  For now, we kill the host process, which is safe
          // but slightly misses the point of fault isolation.
        }
        HostServiceResponse reply{1, 0};
        Sandbox s;
        {
          decltype(ranges)::iterator r;
          std::lock_guard g(m);
          r = ranges.find(fd);
          if (r == ranges.end())
          {
            continue;
          }
          s = r->second.second;
        }
        auto is_large_sizeclass = [](auto large_size) {
          return (large_size < snmalloc::NUM_LARGE_CLASSES);
        };
        // No default so we get range checking.  Fallthrough returns the error
        // result.
        switch (rpc.kind)
        {
          case MemoryProviderPushLargeStack:
          {
            // This may truncate, but only if the sandbox is doing
            // something bad.  We'll catch that on the range check (or get
            // some in-sandbox corruption that the sandbox could do
            // anyway), so don't bother checking it now.
            void* base = reinterpret_cast<void*>(rpc.arg0);
            uint8_t large_size = static_cast<uint8_t>(rpc.arg1);
            if (
              !is_large_sizeclass(large_size) ||
              !s.memory_provider->contains(
                base, snmalloc::large_sizeclass_to_size(large_size)))
            {
              break;
            }
            s.memory_provider->push_large_stack(
              static_cast<snmalloc::Largeslab*>(base), large_size);
            reply = {0, 0};
            break;
          }
          case MemoryProviderPopLargeStack:
          {
            uint8_t large_size = static_cast<uint8_t>(rpc.arg1);
            if (!is_large_sizeclass(large_size))
            {
              break;
            }
            reply = {0,
                     reinterpret_cast<uintptr_t>(
                       s.memory_provider->pop_large_stack(large_size))};
            break;
          }
          case MemoryProviderReserve:
          {
            uint8_t large_size = static_cast<uint8_t>(rpc.arg0);
            if (!is_large_sizeclass(large_size))
            {
              break;
            }
            reply = {0,
                     reinterpret_cast<uintptr_t>(
                       s.memory_provider->template reserve<true>(large_size))};
            break;
          }
          case ChunkMapSet:
          case ChunkMapSetRange:
          case ChunkMapClearRange:
          {
            reply.error = !validate_and_insert(s, rpc);
            break;
          }
        }
        ssize_t remainder = sizeof(HostServiceResponse);
        char* buffer = reinterpret_cast<char*>(&reply);
        while (remainder > 0)
        {
          ssize_t ret = write(fd, buffer, remainder);
          if (ret < 0)
          {
            if ((errno == EAGAIN) || ((errno == EINTR)))
            {
              continue;
            }
            // All other errno errors may be the result of the child doing
            // something wrong.  Let the child fail here.
            break;
          }
          remainder -= ret;
          buffer += ret;
        }
      }
      err(1, "Waiting for pagetable updates failed");
    }
    /**
     * Validate a request from the sandbox to update a pagemap and insert it if
     * allowed.  The `sender` parameter is the file descriptor over which the
     * message was sent. The `position` parameter is the address of the memory
     * for which the corresponding pagemap entry is to be updated.  For the
     * update to succeed, this must be within the range owned by the sandbox
     * identified by the sending socket.  The last two parameters indicate
     * whether this is a large allocation (one that spans multiple pagemap
     * entries) and the value.  If `isBig` is 0, then this is a simple update
     * of a single pagemap entry (typically a slab), specified in the `value`
     * parameter.  Otherwise, `value` is the base-2 logarithm of the size,
     * which is either being set or cleared (`isBig` values of 1 and 2,
     * respectively).
     */
    bool validate_and_insert(Sandbox& s, HostServiceRequest rpc)
    {
      void* address = reinterpret_cast<void*>(rpc.arg0);
      snmalloc::ChunkmapPagemap& cpm = snmalloc::GlobalPagemap::pagemap();
      size_t index = cpm.index_for_address(rpc.arg0);
      size_t entries = 1;
      bool safe = true;
      auto check_large_update = [&]() {
        size_t alloc_size = (1ULL << rpc.arg1);
        entries = alloc_size / snmalloc::SUPERSLAB_SIZE;
        return s.memory_provider->contains(address, alloc_size);
      };
      switch (rpc.kind)
      {
        default:
          // Should be unreachable
          SANDBOX_DEBUG_INVARIANT(false, "Invalid RPC kind: {}", rpc.kind);
          __builtin_unreachable();
          break;
        case ChunkMapSet:
          if ((safe = s.memory_provider->contains(
                 address, snmalloc::SUPERSLAB_SIZE)))
          {
            cpm.set(rpc.arg0, rpc.arg1);
          }
          break;
        case ChunkMapSetRange:
          if ((safe = check_large_update()))
          {
            pm.set_large_size(address, 1ULL << rpc.arg1);
          }
          break;
        case ChunkMapClearRange:
          if ((safe = check_large_update()))
          {
            pm.clear_large_size(address, 1ULL << rpc.arg1);
          }
      }
      if (safe)
      {
        for (size_t i = 0; i < entries; i++)
        {
          s.shared_page[index + i] =
            pm.get(pointer_offset(address, i * snmalloc::SUPERSLAB_SIZE));
        }
      }
      return safe;
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
    void add_range(
      SharedMemoryProvider* memory_provider,
      platform::Handle&& socket,
      platform::SharedMemoryMap& page)
    {
      {
        std::lock_guard g(m);
        register_fd(socket);
        platform::handle_t socket_fd = socket.fd;
        ranges[socket_fd] = std::make_pair(
          std::move(socket),
          Sandbox{memory_provider, static_cast<uint8_t*>(page.get_base())});
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
   * Adaptor for allocators in the shared region to update the pagemap.
   * These treat the global pagemap in the process as canonical but also
   * update the pagemap in the child whenever the parent allocates within the
   * shared region.
   */
  struct SharedPagemapAdaptor
  {
    /**
     * Interface to the global pagemap.  Used to update the global pagemap and
     * to query values to propagate to the child process.
     */
    snmalloc::DefaultChunkMap<> global_pagemap;
    /**
     * The page in the child process that will be mapped into its pagemap.  Any
     * slab allocations by the parent must be propagated into this page.
     */
    uint8_t* shared_page;

    /**
     * Constructor.  Takes a shared pagemap page that this adaptor will update
     * in addition to updating the global pagemap.
     */
    SharedPagemapAdaptor(uint8_t* p) : shared_page(p) {}

    /**
     * Update the child, propagating `entries` entries from the global pagemap
     * into the shared pagemap region.
     */
    void update_child(uintptr_t p, size_t entries = 1)
    {
      snmalloc::ChunkmapPagemap& cpm = snmalloc::GlobalPagemap::pagemap();
      size_t index = cpm.index_for_address(p);
      for (size_t i = 0; i < entries; i++)
      {
        shared_page[index + i] =
          global_pagemap.get(p + (i * snmalloc::SUPERSLAB_SIZE));
      }
    }
    /**
     * Accessor.  We treat the global pagemap as canonical, so only look values
     * up here.
     */
    uint8_t get(uintptr_t p)
    {
      return global_pagemap.get(p);
    }
    /**
     * Set a superslab entry in the pagemap.  Inserts it into the global
     * pagemap and then propagates to the child.
     */
    void set_slab(snmalloc::Superslab* slab)
    {
      global_pagemap.set_slab(slab);
      update_child(reinterpret_cast<uintptr_t>(slab));
    }
    /**
     * Clear a superslab entry in the pagemap.  Removes it from the global
     * pagemap and then propagates to the child.
     */
    void clear_slab(snmalloc::Superslab* slab)
    {
      global_pagemap.clear_slab(slab);
      update_child(reinterpret_cast<uintptr_t>(slab));
    }
    /**
     * Clear a medium slab entry in the pagemap.  Removes it from the global
     * pagemap and then propagates to the child.
     */
    void clear_slab(snmalloc::Mediumslab* slab)
    {
      global_pagemap.clear_slab(slab);
      update_child(reinterpret_cast<uintptr_t>(slab));
    }
    /**
     * Set a medium slab entry in the pagemap.  Inserts it into the global
     * pagemap and then propagates to the child.
     */
    void set_slab(snmalloc::Mediumslab* slab)
    {
      global_pagemap.set_slab(slab);
      update_child(reinterpret_cast<uintptr_t>(slab));
    }
    /**
     * Set a large entry in the pagemap.  Inserts it into the global
     * pagemap and then propagates to the child.
     */
    void set_large_size(void* p, size_t size)
    {
      global_pagemap.set_large_size(p, size);
      size_t entries = size / snmalloc::SUPERSLAB_SIZE;
      update_child(reinterpret_cast<uintptr_t>(p), entries);
    }
    /**
     * Clear a large entry in the pagemap.  Removes it from the global
     * pagemap and then propagates to the child.
     */
    void clear_large_size(void* p, size_t size)
    {
      global_pagemap.clear_large_size(p, size);
      size_t entries = size / snmalloc::SUPERSLAB_SIZE;
      update_child(reinterpret_cast<uintptr_t>(p), entries);
    }
  };

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
        return return_int(-EINVAL);
      }
      Path path{raw_path.get()};
      path.canonicalise();
      auto allowed = vfs.lookup_file(path);
      if (!allowed.has_value())
      {
        return return_int(-ENOENT);
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
          return return_int(-EINVAL);
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
     * Handle a request.
     */
    void handle(Library& lib)
    {
      CallbackRequest req;
      platform::Handle in_fd;
      // FIXME: This should not block, but it can if the sandbox doesn't write
      // anything into the socket.
      socket.receive(&req, sizeof(req), in_fd);

      CallbackHandlerBase::Result ret;
      if (req.kind < handlers.size())
      {
        ret = handlers[req.kind]->invoke(lib, req);
      }
      socket.send(&ret.integer, sizeof(ret.integer), ret.handle);
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
    };
  };

  ExportedFileTree& Library::filetree()
  {
    return callback_dispatcher->vfs;
  }

  int Library::register_callback(
    std::unique_ptr<CallbackHandlerBase>&& callback)
  {
    return callback_dispatcher->register_callback(std::move(callback));
  }

  Library::~Library()
  {
    wait_for_child_exit();
    shared_mem->destroy();
  }

  void Library::start_child(
    const char* library_name,
    const char* librunnerpath,
    const void* sharedmem_addr,
    platform::Handle& pagemap_mem,
    platform::Handle&& malloc_rpc_socket,
    platform::Handle&& fd_socket)
  {
    static const int last_fd = OtherLibraries;
    auto move_fd = [](int x) {
      assert(x >= 0);
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
    pagemap_mem = move_fd(pagemap_mem.take());
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
    dup2(pagemap_mem.take(), PageMapPage);
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
    shared_pagemap(snmalloc::bits::next_pow2_bits(snmalloc::OS_PAGE_SIZE)),
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
      &memory_provider, std::move(malloc_rpc_sockets.first), shared_pagemap);
    // Construct a UNIX domain socket.  This will eventually be used to send
    // file descriptors from the parent to the child, but isn't yet.
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
        shared_pagemap.get_handle(),
        std::move(malloc_rpc_sockets.second),
        std::move(socks.second));
    });
    callback_dispatcher->socket = std::move(socks.first);
    // Allocate an allocator in the shared memory region.
    allocator = new SharedAlloc(
      memory_provider,
      SharedPagemapAdaptor(static_cast<uint8_t*>(shared_pagemap.get_base())),
      &shared_mem->allocator_state);
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
