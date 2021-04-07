// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
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
#include "platform/platform.h"
#include "sandbox_fd_numbers.h"

#include <snmalloc.h>

#ifndef SANDBOX_PAGEMAP
#  ifdef SNMALLOC_DEFAULT_PAGEMAP
#    define SANDBOX_PAGEMAP SNMALLOC_DEFAULT_PAGEMAP
#  else
#    define SANDBOX_PAGEMAP snmalloc::SuperslabMap
#  endif
#endif

namespace snmalloc
{
  template<class T>
  class MemoryProviderStateMixin;
  template<class T>
  class PALPlainMixin;
  template<
    bool (*NeedsInitialisation)(void*),
    void* (*InitThreadAllocator)(function_ref<void*(void*)>),
    class MemoryProvider,
    class ChunkMap,
    bool IsQueueInline>
  class Allocator;
  template<typename T>
  struct SuperslabMap;
  void* no_replacement(void*);
}

namespace sandbox
{
  struct SharedMemoryRegion;
  struct SharedPagemapAdaptor;
  struct MemoryProviderBumpPointerState;
  class CallbackDispatcher;
  class ExportedFileTree;
  struct CallbackHandlerBase;

  /**
   * An snmalloc Platform Abstraction Layer (PAL) that cannot be used to
   * allocate memory.  This is used for sandboxes, where all memory comes from a
   * pre-defined shared region.
   */
  using NoOpPal = snmalloc::PALNoAlloc<snmalloc::DefaultPal>;
  using snmalloc::pointer_offset;

  /**
   * The memory provider for the shared region.  This manages a single
   * contiguous address range.
   */
  class SharedMemoryProvider
  : public snmalloc::MemoryProviderStateMixin<NoOpPal>
  {
    /**
     * Base address of the shared memory region.
     */
    void* base;

    /**
     * Top of the shared memory region.
     */
    void* top;

  public:
    /**
     * Constructor.  Takes the memory range allocated for the sandbox heap as
     * arguments.
     */
    SharedMemoryProvider(void* base_address, size_t length)
    : MemoryProviderStateMixin<NoOpPal>(base_address, length),
      base(base_address),
      top(pointer_offset(base, length))
    {}

    /**
     * Predicate to test whether an object of size `sz` starting at `ptr`
     * is within the region managed by this memory provider.
     */
    bool contains(const void* ptr, size_t sz)
    {
      // We shouldn't need the const cast here, but pointer_offset doesn't
      // correctly handle const pointers yet.
      return (ptr >= base) &&
        (pointer_offset(const_cast<void*>(ptr), sz) < top);
    }

    /**
     * Return the top of the sandbox.
     */
    void* top_address()
    {
      return top;
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

    SNMALLOC_FAST_PATH static bool needs_initialisation(void*)
    {
      return false;
    }
    SNMALLOC_FAST_PATH static void*
      init_thread_allocator(snmalloc::function_ref<void*(void*)>)
    {
      return nullptr;
    }
    /**
     * The type of the allocator that allocates within the shared region from
     * outside.  This has an out-of-line message queue, allocated in the shared
     * memory region, and updates both the child and parent views of the
     * pagemap when allocating new slabs.
     */
    using SharedAlloc = snmalloc::Allocator<
      needs_initialisation,
      init_thread_allocator,
      SharedMemoryProvider,
      SharedPagemapAdaptor,
      false>;
    /**
     * The allocator used for allocating memory inside this sandbox.
     */
    SharedAlloc* allocator;
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
     * A flag indicating whether the child has exited.  This is updated when a
     * message send fails.
     */
    bool child_exited = false;
    /**
     * The exit code of the child.  This is set when the child process exits.
     */
    int child_status;

    /**
     * The shared memory object that contains the child process's heap.
     */
    platform::SharedMemoryMap shm;

    /**
     * The shared pagemap page.
     */
    platform::SharedMemoryMap shared_pagemap;

    /**
     * The (trusted) memory provider that is used to allocate large regions to
     * memory allocators.  This is used directly from outside of the sandbox
     * and via an RPC mechanism that checks arguments from inside.
     */
    SharedMemoryProvider memory_provider;

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
     * The `pagemap_mem` parameter is the file descriptor for the pagemap
     * shared page.
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
      platform::Handle& pagemap_mem,
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
    T* alloc(size_t count)
    {
      static_assert(
        std::is_standard_layout_v<T> && std::is_trivial_v<T>,
        "Arrays allocated in sandboxes must be POD types");
      T* array = static_cast<T*>(alloc_in_sandbox(sizeof(T), count));
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
    T* alloc()
    {
      static_assert(
        std::is_standard_layout_v<T> && std::is_trivial_v<T>,
        "Arrays allocated in sandboxes must be POD types");
      T* array = static_cast<T*>(alloc_in_sandbox(sizeof(T), Count));
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
    T* alloc(Args&&... args)
    {
      static_assert(
        !std::is_polymorphic_v<T>,
        "Classes with vtables cannot be safely allocated in sandboxes from "
        "outside (nor can virtual functions be safely called).");
      return new (alloc_in_sandbox(sizeof(T), 1))
        T(std::forward<Args>(args)...);
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
      char* ptr = alloc<char>(len);
      memcpy(ptr, str, len);
      return ptr;
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
