// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <mutex>
#include <string.h>
#include <tuple>
#include <vector>
#ifdef __unix__
#  include <fcntl.h>
#  include <pthread.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

#include "helpers.h"
#include "netpolicy.h"
#include "platform/platform.h"
#include "sandbox_fd_numbers.h"
#include "sandbox_meta_entry.h"
#include "shared_memory_region.h"

namespace sandbox
{
  class CallbackDispatcher;
  class ExportedFileTree;
  struct CallbackHandlerBase;
  class Library;

  /**
   * An snmalloc Platform Abstraction Layer (PAL) that cannot be used to
   * allocate memory.  This is used for sandboxes, where all memory comes from a
   * pre-defined shared region.
   */
  using NoOpPal = snmalloc::PALNoAlloc<snmalloc::DefaultPal>;

  using snmalloc::pointer_diff;
  using snmalloc::pointer_offset;

  /**
   * Snmalloc back-end structure for shared memory allocations.  This defines
   * how snmalloc will interact with allocations that are per-sandbox.
   *
   * Globals used by the snmalloc instances that allocate sandbox memory from
   * the outside.
   */
  struct SharedAllocConfig : public snmalloc::CommonConfig
  {
    /**
     * Metadata for a slab.  This back end does not store any backend-specific
     * metadata here.
     */
    using SlabMetadata = snmalloc::FrontendSlabMetadata;

    /**
     * The type of a pagemap that spans the entire address space.
     *
     * This is used for all compartments.  Currently, this is not the same
     * pagemap that the parent process uses for any non-shared allocations.
     * This could be changed in environments (such as Verona) that can
     * guarantee that in-compartment pointers are never freed with the global
     * `free` or if we have a default `Alloc` that checks for this case
     * (perhaps with a flag in the metadata entry indicating whether the slab
     * is compartment-owned).
     */
    class Pagemap
    {
    public:
      /**
       * Pagemap entry.
       */
      using Entry = SandboxMetaEntry;

    private:
      /**
       * Concrete instance of a pagemap.  This is updated only with
       * `pagemap_lock` held.
       *
       * This pagemap spans the entire address space (i.e. does not use the
       * fixed-range option) because it covers all sandboxes.  Each sandbox is
       * (currently) a fixed range within the global address space.
       */
      inline static snmalloc::FlatPagemap<
        snmalloc::MIN_CHUNK_BITS,
        Entry,
        NoOpPal,
        /*fixed range*/ false>
        pagemap;

      /**
       * Shared memory object backing the pagemap.  Every compartment has a
       * read-only view of this that is mapped into its address space on start.
       *
       * This is a `unique_ptr` because otherwise this is not discarded in
       * debug builds of the library runner and so ends up trying to initialise
       * a new shared memory (which is not needed) in the child.  This doesn't
       * matter too much, except that on GNU/Linux (and possibly other Linux
       * variants) `shm_open` forwards to `open`, which then tries to do the
       * `open` callback, before the callback mechanism is initialised.
       */
      inline static std::unique_ptr<platform::SharedMemoryMap> pagemap_mem;

      /**
       * Mutex that must be held while writing to the pagemap.
       *
       * This is a recursive mutex because it must be held during destruction
       * of a `Library` and by each call to `set_meta_data` in that destructor.
       */
      inline static std::recursive_mutex pagemap_lock;

    public:
      /**
       * Helper to get a reference to the pagemap and a lock guard.  This is the
       * only way of accessing the pagemap outside of this class.  It should be
       * used as:
       *
       * ```c++
       * auto [g, pm] = Pagemap::get_pagemap_writeable();
       * ```
       *
       * It is then safe to update `pm` for as long as `g` remains in scope.
       * By default, these have the same lifetime.
       */
      std::tuple<
        std::unique_lock<decltype(pagemap_lock)>,
        decltype(pagemap)&> static get_pagemap_writeable()
      {
        std::unique_lock g(pagemap_lock);
        return {std::move(g), pagemap};
      }

      /**
       * Return a reference to the handle used for the pagemap.
       */
      static auto& get_pagemap_handle()
      {
        return pagemap_mem->get_handle();
      }

      /**
       * Retrieve a mutable reference to the metadata entry.  This is used only
       * in the large buddy allocator.
       */
      template<bool potentially_out_of_range = false>
      static Entry& get_metaentry_mut(snmalloc::address_t p)
      {
        return pagemap.template get_mut<potentially_out_of_range>(p);
      }

      /**
       * Look up the metadata entry for an address.  This is called by snmalloc
       * on the deallocation path to determine who owns the memory and, if it's
       * the deallocating allocator, where to find the metadata.
       */
      template<bool potentially_out_of_range = false>
      static const Entry& get_metaentry(snmalloc::address_t p)
      {
        return pagemap.template get<potentially_out_of_range>(p);
      }

      /**
       * Sets metadata in the shared pagemap.  This assumes callers are trusted
       * and does not validate the metadata.  This is called only by the trusted
       * allocator, the RPC thread updating the pagemap on behalf of a child
       * will write to the pagemap directly.
       *
       * In the case of a conflict over ownership, the caller of this always
       * wins.  The RPC handler will check (with the lock held) if a `MetaEntry`
       * identifies an out-of-sandbox allocator as the owner already and refuse
       * to install a new version if it does.  This method, in contrast, will
       * update the pagemap unconditionally.  This means that if the update by
       * the trusted allocator is ordered first (by the lock) then the child
       * will not install an update and if it is ordered second then it will
       * overwrite the child's entry.  This means that the trusted allocator can
       * end up with surprising values in its message queue (which it must
       * protect against anyway because its message queue is writeable by
       * untrusted code) but we cannot leak out-of-sandbox metaslabs as a result
       * of activity by the child.
       */
      static void
      set_metaentry(snmalloc::address_t p, size_t size, const Entry& t)
      {
        auto [g, pm] = get_pagemap_writeable();
        for (snmalloc::address_t a = p; a < p + size;
             a += snmalloc::MIN_CHUNK_SIZE)
        {
          pm.set(a, t);
        }
      }

      /**
       * Initialise the pagemap.  This must be called precisely once, from
       * `SharedAllocConfig::ensure_init`.
       */
      static void init()
      {
        pagemap_mem = std::make_unique<platform::SharedMemoryMap>(
          static_cast<uint8_t>(snmalloc::bits::next_pow2_bits_const(
            decltype(pagemap)::required_size())));
        assert(pagemap_mem->get_base());
        pagemap.init(static_cast<Entry*>((pagemap_mem->get_base())));
      }

      /**
       * Ensure that a range of the pagemap is useable.
       */
      static void register_range(snmalloc::address_t p, size_t sz)
      {
        pagemap.register_range(p, sz);
      }
    };

    /**
     * The memory provider for the shared region.  This manages a single
     * contiguous address range.  This class is used both by the sandboxing code
     * and directly by snmalloc, which holds a reference to an instance of this
     * in the allocator and uses it to allocate memory.  Snmalloc does not call
     * any methods on this class but will pass it as the local-state parameter
     * to the back-end functions that allocate memory.
     *
     * This class must be thread safe: both the RPC thread that services
     * requests from the child process and the allocator that is owned by the
     * sandboxed library object will access it concurrently.  There are two
     * pieces of shared mutable state:
     *
     *  - The shared address-space manager.  This is protected by a flag lock
     *    internally.
     *  - The chunk allocator state.  This contains a fixed-size array of
     *    multi-producer, multi-consumer stacks and so is safe to access from
     *    both threads.
     */
    class LocalState
    {
      /**
       * Base address of the shared memory region.
       */
      void* base;

      /**
       * Top of the shared memory region.
       */
      void* top;

      /**
       * The class that manages blocks of memory for this region.  This is
       * filled with the memory assigned to a specific sandbox on start.
       */
      snmalloc::LargeBuddyRange<
        snmalloc::EmptyRange,
        snmalloc::bits::BITS - 1,
        snmalloc::bits::BITS - 1,
        Pagemap>
        memory;

      /**
       * Lock protecting the range.
       */
      std::mutex lock;

    public:
      /**
       * Helper to receive a reference to the range that can allocate /
       * deallocate memory while holding its lock.
       */
      std::tuple<std::unique_lock<decltype(lock)>, decltype(memory)&>
      get_memory()
      {
        std::unique_lock g{lock};
        return {std::move(g), memory};
      }

      /**
       * Constructor.  Takes the memory range allocated for the sandbox heap as
       * arguments.  This class takes responsibility for allocating memory from
       * the provided range.  Nothing should access any of the memory in this
       * range without first calling the `reserve` method on this class to
       * acquire a chunk.
       */
      LocalState(void* start, size_t size);

      /**
       * Predicate to test whether an object of size `sz` starting at `ptr`
       * is within the region managed by this memory provider.
       *
       * Note that this is smaller than the shared memory object.  The shared
       * memory object starts with a fixed layout.  This region is still
       * treated as untrusted (the child process can modify it) but nothing in
       * that part should ever be passed directly to the parent.
       */
      bool contains(const void* ptr, size_t sz)
      {
        return (ptr >= pointer_offset(base, sizeof(SharedMemoryRegion))) &&
          (top > ptr) && (pointer_diff(ptr, top) >= sz);
      }

      /**
       * Return the top of the sandbox.
       */
      void* top_address()
      {
        return top;
      }

      /**
       * Return the top of the sandbox.
       */
      void* base_address()
      {
        return base;
      }
    };

    /**
     * The PAL that snmalloc will use for this back end.  This is used only by
     * snmalloc.
     */
    using Pal = NoOpPal;

    /**
     * Allocate a chunk, its associated metaslab, and install its metadata
     * entry in the pagemap.  This allocates the chunk in the sandbox-shared
     * memory region and the metaslab in host-owned memory.  This means that
     * all metadata associated with an allocation from outside is inaccessible
     * by the sandbox and does not need to be validated.
     */
    static std::pair<snmalloc::capptr::Chunk<void>, SlabMetadata*>
    alloc_chunk(LocalState& local_state, size_t size, uintptr_t ras)
    {
      snmalloc::capptr::Chunk<void> chunk;
      {
        auto [g, m] = local_state.get_memory();
        chunk = m.alloc_range(size);
      }
      if (chunk == nullptr)
      {
        return {nullptr, nullptr};
      }
      auto* meta = new SlabMetadata();
      Pagemap::Entry t(meta, ras);
      Pagemap::set_metaentry(address_cast(chunk), size, t);

      chunk =
        snmalloc::Aal::capptr_bound<void, snmalloc::capptr::bounds::Chunk>(
          chunk, size);
      return {chunk, meta};
    }

    /**
     * Free a chunk of memory in the shared memory region.  This is used
     * directly by `dealloc_chunk` and also to handle the RPC call from the
     * child to deallocate memory.
     */
    static void dealloc_range(
      LocalState& local_state, snmalloc::capptr::Chunk<void> base, size_t size)
    {
      Pagemap::Entry t;
      t.claim_for_backend();
      Pagemap::set_metaentry(base.unsafe_uintptr(), size, t);
      auto [g, m] = local_state.get_memory();

      m.dealloc_range(base, size);
    }

    /**
     * Deallocate a chunk and its associated metadata.
     */
    static void dealloc_chunk(
      LocalState& local_state,
      SlabMetadata& metadata,
      snmalloc::capptr::Alloc<void> base,
      size_t size)
    {
      snmalloc::capptr::Chunk<void> chunk{base.unsafe_ptr()};
      dealloc_range(local_state, chunk, size);
      delete &metadata;
    }

    /**
     * Allocate metadata.  Metadata is stored outside of the sandbox and so
     * this is just allocated with the normal malloc.
     *
     * This has a single concrete specialisation, to allocate metaslabs.  Any
     * modifications to snmalloc that try to allocate other types though this
     * interface will cause linker failures, allowing us to check whether they
     * need to be allocated in the shared memory region or not.
     *
     * Note that there is no specialisation of this function to allocate
     * allocators.  This would be used by the pool allocator functionality in
     * snmalloc.  We should have exactly one shared allocator per sandbox and
     * so we don't use the pool allocator and will get a link failure if we
     * accidentally do.
     */
    template<typename T>
    static snmalloc::capptr::Chunk<void>
    alloc_meta_data(LocalState*, size_t size);

    /**
     * Options for configuring snmalloc.  This allocator is almost the exact
     * opposite of the default.  There is only one of them per sandbox, they
     * aren't per-thread, they're allocated and deallocated by the sandbox
     * library.  The untrusted code has write access to the message queues and
     * so the queue heads are not trusted.
     */
    constexpr static snmalloc::Flags Options{
      .IsQueueInline = false,
      .CoreAllocOwnsLocalState = false,
      .CoreAllocIsPoolAllocated = false,
      .LocalAllocSupportsLazyInit = false,
      .QueueHeadsAreTame = false,
      .HasDomesticate = true,
    };

    /**
     * 'Domesticate' a pointer.  This takes a pointer to something that we've
     * read from sandbox-controlled memory and validates whether it can be used
     * for the specified type.  If it does not come from the sandbox identified
     * by `ls` then this return a null pointer.
     */
    template<typename T, SNMALLOC_CONCEPT(snmalloc::capptr::ConceptBound) B>
    static auto capptr_domesticate(LocalState* ls, snmalloc::CapPtr<T, B> p)
    {
      // If we know the size that we're being asked for then use it in the
      // check.  For deallocated objects the type is `void` and the caller
      // will check that this is part of a valid allocation, we'll check that
      // it's sufficiently large to store a freelist entry.
      using ObjType = std::conditional_t<std::is_same_v<T, void>, void*, T>;
      T* unsafe_ptr = ls->contains(p.unsafe_ptr(), sizeof(ObjType)) ?
        p.unsafe_ptr() :
        nullptr;
      using Tame = typename B::template with_wildness<
        snmalloc::capptr::dimension::Wildness::Tame>;
      return snmalloc::CapPtr<T, Tame>(unsafe_ptr);
    }

  private:
    friend class LocalState;

    /**
     * Construct the global state object.  This allocates the huge region for
     * the pagemap (512 GiB currently on most platforms, subject to change) and
     * initialises the pagemap object to point to it.
     *
     * This does not use snmalloc's lazy initialisation logic because we need
     * to create the shared memory *before* we create any allocators.  It is
     * called in LocalState's constructor.
     */
    inline static void ensure_initialised()
    {
      static std::atomic<bool> isInitialised = false;
      if (isInitialised)
      {
        return;
      }
      isInitialised = true;
      Pagemap::init();
    }
  };

  /**
   * Class encapsulating an instance of a shared library in a sandbox.
   * Instances of this class will create a sandbox and load a specified library
   * into it, but are useless in isolation.  The `Function` class
   * provides a wrapper for calling an exported function in the specified
   * library.
   */
  class Library
  {
    /**
     * `handle_t` is the type used by the OS for handles to operating system
     * resources.  On *NIX systems, file descriptors are represented as
     * `int`s.
     */
    using handle_t = platform::handle_t;

    /**
     * The type of the allocator that allocates within the shared region from
     * outside.  This has an out-of-line message queue, allocated in the shared
     * memory region, and updates both the child and parent views of the
     * pagemap when allocating new slabs.
     */
    using SharedAlloc = snmalloc::LocalAllocator<SharedAllocConfig>;

    /**
     * A pointer to the core allocator.  Each snmalloc allocator is a pair of a
     * core allocator and a local allocator.  The former provides the slow-path
     * operations and is managed by the latter, which provides fast-path
     * operations.
     *
     * This is a `unique_ptr` because it can't be constructed until things that
     * are created in the middle of the constructor are available as
     * constructor parameters and C++ requires all objects to be constructed on
     * entry to the constructor.
     */
    std::unique_ptr<snmalloc::CoreAllocator<SharedAllocConfig>> core_alloc;

    /**
     * The allocator used for allocating memory inside this sandbox.
     */
    std::unique_ptr<SharedAlloc> allocator;

    /**
     * The handle to the socket that is used to pass file descriptors to the
     * sandboxed process.
     */
    platform::SocketPair::Socket socket;

    /**
     * The platform-specific child process.
     */
    std::unique_ptr<platform::ChildProcess> child_proc;
    /**
     * A pointer to the shared-memory region.  The start of this is structured,
     * the rest is an untyped region of memory that can be used to allocate
     * slabs and large objects.
     */
    struct SharedMemoryRegion* shared_mem;
    /**
     * The first unused vtable entry.  When a sandboxed library is created, all
     * of the functions register themselves at a specific index.
     *
     * The first vtable entry is reserved for the function that returns the
     * type encoding of a specific vtable entry.  This is used to ensure that
     * the child and parent agree on the type signatures of all exported
     * functions.
     */
    int last_vtable_entry = 1;

    /**
     * The exit code of the child.  This is set when the child process exits.
     */
    int child_status;

    /**
     * The shared memory object that contains the child process's heap.
     */
    platform::SharedMemoryMap shm;

    /**
     * The (trusted) memory provider that is used to allocate large regions to
     * memory allocators.  This is used directly from outside of the sandbox
     * and via an RPC mechanism that checks arguments from inside.
     */
    SharedAllocConfig::LocalState memory_provider;

    /**
     * Allocate some memory in the sandbox.  Returns `nullptr` if the
     * allocation failed.
     */
    void* alloc_in_sandbox(size_t bytes, size_t count);

    /**
     * Deallocate an allocation in the sandbox.
     */
    void dealloc_in_sandbox(void* ptr);

    /**
     * Start the child process.  On *NIX systems, this can be called within a
     * vfork context and so must not allocate or modify memory on the heap, or
     * read from the heap in a way that is not safe in the presence of
     * concurrency.
     *
     * The `library_name` parameter is the path to the library that should be
     * launched.
     *
     * The `librunnerpath` parameter is the full path to the `library_runner`
     * binary that runs as the child process, loads the library, and so on.
     *
     * The `sharedmem_addr` is the address at which the shared memory region
     * should be mapped in the child.
     *
     * The `pagemap_mem` parameter is the file descriptor for the shared memory
     * backing the pagemap that is used by all sandboxes.  This must not be
     * closed in the parent process.
     *
     * The `pagemap_pipe` parameter is the file descriptor for the pipe used to
     * send pagemap updates from the child to the parent.
     *
     * The `fd_socket` parameter is the file descriptor for a socket that can
     * be used to send file descriptors to the child process.
     */
    [[noreturn]] void start_child(
      const char* library_name,
      const char* librunnerpath,
      const void* sharedmem_addr,
      const platform::Handle& pagemap_mem,
      platform::Handle&& pagemap_pipe,
      platform::Handle&& fd_socket);

    /**
     * The delegate that handles callbacks for this sandbox.
     */
    std::unique_ptr<CallbackDispatcher> callback_dispatcher;

  public:
    /**
     * Returns the next vtable entry to use, incrementing the counter so
     * subsequent calls will always return a fresh value.
     */
    int next_vtable_entry()
    {
      return last_vtable_entry++;
    }
    /**
     * Destructor.  Cleans up the shared memory region.
     *
     * Note that all pointers into memory owned by the library are invalid
     * after this has been deallocated.
     */
    ~Library();
    /**
     * Constructor.  Creates a new sandboxed instance of the library named by
     * `library_name`, with the heap size specified in GiBs.
     */
    Library(const char* library_name, size_t heap_size_in_GiBs = 1);
    /**
     * Allocate space for an array of `count` instances of `T`.  Objects in the
     * array will be default constructed.
     *
     * Only POD types may be allocated in the sandbox - anything with a vtable
     * would have its vtable incorrectly initialised.
     */
    template<typename T>
    std::optional<T*> alloc(size_t count)
    {
      static_assert(
        std::is_standard_layout_v<T> && std::is_trivial_v<T>,
        "Arrays allocated in sandboxes must be POD types");
      T* array = static_cast<T*>(alloc_in_sandbox(sizeof(T), count));
      if (array == nullptr)
      {
        return std::nullopt;
      }
      for (size_t i = 0; i < count; i++)
      {
        new (&array[i]) T();
      }
      return array;
    }

    /**
     * Returns the filesystem abstraction exported to this sandbox.
     */
    ExportedFileTree& filetree();

    /**
     * Returns the network access policy for this sandbox.
     */
    NetworkPolicy& network_policy();

    /**
     * Register a handler for a callback from this sandbox.  The return value
     * the index of this that should be passed to the `invoke_user_callback`
     * function.
     */
    int register_callback(std::unique_ptr<CallbackHandlerBase>&&);

    /**
     * Allocate space for a fixed-sized array of `Count` instances of `T`.
     * Objects in the array will be default constructed.
     *
     * Only POD types may be allocated in the sandbox - anything with a vtable
     * would have its vtable incorrectly initialised.
     */
    template<typename T, size_t Count>
    std::optional<T*> alloc()
    {
      static_assert(
        std::is_standard_layout_v<T> && std::is_trivial_v<T>,
        "Arrays allocated in sandboxes must be POD types");
      T* array = static_cast<T*>(alloc_in_sandbox(sizeof(T), Count));
      if (array == nullptr)
      {
        return std::nullopt;
      }
      for (size_t i = 0; i < Count; i++)
      {
        new (&array[i]) T();
      }
      return array;
    }
    /**
     * Allocate an object in the sandbox and call its constructor with the
     * specified arguments.
     *
     * Only types without vtables may be allocated in the sandbox - anything
     * with a vtable would have its vtable incorrectly initialised.
     */
    template<typename T, typename... Args>
    std::optional<T*> alloc(Args&&... args)
    {
      static_assert(
        !std::is_polymorphic_v<T>,
        "Classes with vtables cannot be safely allocated in sandboxes from "
        "outside (nor can virtual functions be safely called).");
      void* ptr = alloc_in_sandbox(sizeof(T), 1);
      if (ptr == nullptr)
      {
        return std::nullopt;
      }
      return new (ptr) T(std::forward<Args>(args)...);
    }
    /**
     * Free an object allocated in the sandbox.
     */
    template<typename T>
    void free(T* obj)
    {
      dealloc_in_sandbox(static_cast<void*>(obj));
    }
    /**
     * Helper function to copy a string into a sandbox.  The caller is
     * responsible for freeing the returned memory by calling the `free` method
     * on this class.
     */
    char* strdup(const char* str)
    {
      auto len = strlen(str);
      std::optional<char*> ptr = alloc<char>(len);
      if (!ptr)
      {
        return nullptr;
      }
      memcpy(*ptr, str, len);
      return *ptr;
    }

    /**
     * Copy a string out of the sandbox.  It's easy to introduce TOCTOU bugs
     * when you use C strings that are stored in untrusted memory, this
     * provides an easy way of defensively copying them out.
     */
    unique_c_ptr<char> strdup_out(const char* str)
    {
      if (!contains(str, 1))
      {
        return nullptr;
      }
      auto maxlen = static_cast<char*>(memory_provider.top_address()) - str;
      auto len = strnlen(str, maxlen);
      if (len == static_cast<size_t>(maxlen))
      {
        return nullptr;
      }
      unique_c_ptr<char> ptr;
      ptr.reset(static_cast<char*>(malloc(len + 1)));
      memcpy(ptr.get(), str, len);
      ptr.get()[len] = '\0';
      return ptr;
    }

    /**
     * Predicate to test whether an object of size `sz` starting at `ptr`
     * is within this sandbox.
     */
    bool contains(const void* ptr, size_t sz)
    {
      return memory_provider.contains(ptr, sz);
    }

    /**
     * Returns the bounds of the sandbox region managed by snmalloc.
     */
    std::pair<const void*, const void*> sandbox_heap()
    {
      return {memory_provider.base_address(), memory_provider.top_address()};
    }

  private:
    /**
     * Is this the first time that we've invoked a sandbox?  If so, we will
     * need to wait for it to be ready before we invoke it.
     */
    bool is_first_call = true;
    /**
     * Function is allowed to call the following methods in this class.
     */
    template<typename Ret, typename... Args>
    friend class Function;
    /**
     * Sends a message to the child process, containing a vtable index and a
     * pointer to the argument frame (a tuple of arguments and space for the
     * return value).
     */
    void send(int idx, void* ptr);
    /**
     * Instruct the child to exit and block until it does.  The return value is
     * the exit code of the child process.  If the child has already exited,
     * then this return immediately.
     */
    int wait_for_child_exit();
    /**
     * Pool to determine if the child has exited.  This interface is inherently
     * racy: If it returns `false` there is no guarantee that the child hasn't
     * exited immediately after the call.
     */
    bool has_child_exited();

    /**
     * Forcibly terminate the child.
     */
    void terminate();

    /**
     * MemoryServiceProvider manages memory for sandboxes and must be able to
     * access this class's memory provider.
     */
    friend class MemoryServiceProvider;

    /**
     * CallbackDispatcher needs to be able to terminate a running sandbox.
     */
    friend class CallbackDispatcher;
  };

  /**
   * Function to invoke a callback from within a sandbox.  This takes the
   * number of the callback, which must be a number previously returned from
   * `register_callback` on the `Library` that encapsulates the
   * sandbox from which this is being called.
   *
   * The next two arguments specify the data and size.  The size must be
   * non-zero, even if the callback does not need any state.  The `data`
   * argument will be copied to the heap if it is not already there.
   *
   * If the `fd` argument is not negative, the file descriptor will also be
   * passed along with the callback.  There is not currently a mechanism for
   * passing more than one file descriptor to a callback though this would be
   * easy to add if required.
   */
  int invoke_user_callback(int idx, void* data, size_t size, int fd = -1);

}
