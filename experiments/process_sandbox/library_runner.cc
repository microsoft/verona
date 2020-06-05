#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef USE_CAPSICUM
#  include <sys/capsicum.h>
#endif
#include <aal/aal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

using address_t = snmalloc::Aal::address_t;

namespace snmalloc
{
  template<typename T>
  struct SuperslabMap;
  class Superslab;
  class Mediumslab;
}

namespace sandbox
{
  struct ProxyPageMap
  {
    static ProxyPageMap p;
    static ProxyPageMap& pagemap()
    {
      return p;
    }
    void set(uintptr_t p, uint8_t x, uint8_t big);
    static uint8_t get(address_t p);
    static uint8_t get(void* p);
    void set_slab(snmalloc::Superslab* slab);
    void clear_slab(snmalloc::Superslab* slab);
    void clear_slab(snmalloc::Mediumslab* slab);
    void set_slab(snmalloc::Mediumslab* slab);
    void set_large_size(void* p, size_t size);
    void clear_large_size(void* p, size_t size);
  };
}
sandbox::ProxyPageMap sandbox::ProxyPageMap::p;

#define SNMALLOC_DEFAULT_CHUNKMAP sandbox::ProxyPageMap
#define SNMALLOC_USE_THREAD_CLEANUP 1
#include "libsandbox.cc"
#include "override/malloc.cc"
#include "shared.h"

using namespace snmalloc;
using namespace sandbox;

namespace
{
#ifdef __linux__
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

  dlfunc_t dlfunc(void* handle, const char* symbol)
  {
    return (dlfunc_t)dlsym(handle, symbol);
  }
#endif

  int pagemap_socket = -1;
  ExportedLibrary* library;
  void* shared_memory_start = 0;
  void* shared_memory_end = 0;
  char* exported_types(int idx)
  {
    return library->type_encoding(idx);
  }
}
namespace sandbox
{
  void ProxyPageMap::set(uintptr_t p, uint8_t x, uint8_t big = 0)
  {
    if (
      (p >= reinterpret_cast<uintptr_t>(shared_memory_start)) &&
      (p < reinterpret_cast<uintptr_t>(shared_memory_end)))
    {
      assert(pagemap_socket > 0);
      auto msg = static_cast<uint64_t>(p);
      // Make sure that the low 16 bytes are clear
      assert((msg & 0xffff) == 0);
      msg &= ~0xffff;
      msg |= x;
      msg |= big << 8;
      write(pagemap_socket, static_cast<void*>(&msg), sizeof(msg));
      while (GlobalPagemap::pagemap().get(p) != x)
      {
        // write(fileno(stderr), "Child spinning\n", 15);
        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
    }
    else
    {
      GlobalPagemap::pagemap().set(p, x);
    }
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
  void ProxyPageMap::clear_slab(snmalloc::Mediumslab* slab)
  {
    set(reinterpret_cast<uintptr_t>(slab), (size_t)CMNotOurs);
  }
  void ProxyPageMap::set_slab(snmalloc::Mediumslab* slab)
  {
    set(reinterpret_cast<uintptr_t>(slab), (size_t)CMMediumslab);
  }
  void ProxyPageMap::set_large_size(void* p, size_t size)
  {
    size_t size_bits = bits::next_pow2_bits(size);
    if ((p >= shared_memory_start) && (p < shared_memory_end))
    {
      set(reinterpret_cast<uintptr_t>(p), (uint8_t)size_bits, 1);
      return;
    }
    set(reinterpret_cast<uintptr_t>(p), size_bits);
    // Set redirect slide
    uintptr_t ss = (uintptr_t)((size_t)p + SUPERSLAB_SIZE);
    for (size_t i = 0; i < size_bits - SUPERSLAB_BITS; i++)
    {
      size_t run = 1ULL << i;
      GlobalPagemap::pagemap().set_range(
        ss, (uint8_t)(64 + i + SUPERSLAB_BITS), run);
      ss = ss + SUPERSLAB_SIZE * run;
    }
  }
  void ProxyPageMap::clear_large_size(void* p, size_t size)
  {
    if ((p >= shared_memory_start) && (p < shared_memory_end))
    {
      size_t size_bits = bits::next_pow2_bits(size);
      set(reinterpret_cast<uintptr_t>(p), (uint8_t)size_bits, 2);
      return;
    }
    auto range = (size + SUPERSLAB_SIZE - 1) >> SUPERSLAB_BITS;
    GlobalPagemap::pagemap().set_range(
      reinterpret_cast<uintptr_t>(p), CMNotOurs, range);
  }
  class ExportedLibraryPrivate
  {
    friend class ExportedLibrary;
#ifdef __unix__
    using handle_t = int;
#endif
    __attribute__((unused)) handle_t socket_fd;
    struct SharedMemoryRegion* shared_mem;

  public:
    ExportedLibraryPrivate(handle_t sock, SharedMemoryRegion* region)
    : socket_fd(sock), shared_mem(region)
    {}
    void runloop(ExportedLibrary* library)
    {
      while (1)
      {
        shared_mem->wait(true);
        if (shared_mem->should_exit)
        {
          exit(0);
        }
        assert(shared_mem->is_child_executing);
        try
        {
          (*library->functions[shared_mem->function_index])(
            shared_mem->msg_buffer);
        }
        catch (...)
        {
          // FIXME: Report error in some useful way.
          printf("Exception!\n");
        }
        shared_mem->signal(false);
      }
    }
  };
}
char* ExportedLibrary::type_encoding(int idx)
{
  return functions.at(idx)->type_encoding();
}

int main(int, char** argv)
{
#ifdef USE_CAPSICUM
  cap_enter();
#endif
  void* addr = (void*)strtoull(argv[1], nullptr, 0);
  size_t length = strtoull(argv[2], nullptr, 0);
  // fprintf(stderr, "Child starting\n");
  // printf(
  //"Child trying to map fd %d at addr %p (0x%zx)\n", SharedMemRegion, addr,
  // length);
  void* ptr = mmap(
    addr,
    length,
    PROT_READ | PROT_WRITE,
    MAP_FIXED | MAP_ALIGNED(35) | MAP_SHARED | MAP_NOCORE,
    SharedMemRegion,
    0);
  // printf("%p\n", ptr);
  if (ptr == MAP_FAILED)
  {
    err(1, "Mapping shared heap failed");
  }
  // Close the shared memory region file descriptor before we call untrusted
  // code.
  close(SharedMemRegion);

  auto shared = reinterpret_cast<SharedMemoryRegion*>(ptr);
  // Splice the pagemap page inherited from the parent into the pagemap.
  void* pagemap_chunk =
    GlobalPagemap::pagemap().page_for_address(reinterpret_cast<uintptr_t>(ptr));
  munmap(pagemap_chunk, 4096);
  void* shared_pagemap = mmap(
    pagemap_chunk, 4096, PROT_READ, MAP_SHARED | MAP_FIXED, PageMapPage, 0);
  if (shared_pagemap == MAP_FAILED)
  {
    err(1, "Mapping shared pagemap page failed");
  }
  SharedMemoryProvider* global_virtual = &shared->memory_provider;
  shared_memory_start =
    reinterpret_cast<void*>(global_virtual->shared_heap_range_start.load());
  shared_memory_end =
    reinterpret_cast<void*>(global_virtual->shared_heap_range_end);
  assert(shared_pagemap == pagemap_chunk);
  (void)shared_pagemap;
  close(PageMapPage);
  pagemap_socket = PageMapUpdates;

  // Replace the current thread allocator with a new one in the shared region.
  // After this point, all new memory allocations are shared with the parent.
  current_alloc_pool() =
    snmalloc::make_alloc_pool<GlobalVirtual, Alloc>(*global_virtual);
  ThreadAlloc::get_reference() = current_alloc_pool()->acquire();

  void* handle = fdlopen(MainLibrary, RTLD_GLOBAL);
  if (handle == nullptr)
  {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  void (*sandbox_init)(ExportedLibrary*) =
    reinterpret_cast<void (*)(ExportedLibrary*)>(
      dlfunc(handle, "sandbox_init"));
  if (sandbox_init == nullptr)
  {
    fprintf(stderr, "dlfunc failed: %s\n", dlerror());
    return 1;
  }
  ExportedLibraryPrivate* libPrivate;
  libPrivate = new ExportedLibraryPrivate(FDSocket, shared);
  library = new ExportedLibrary();
  library->export_function(exported_types);
  sandbox_init(library);

  libPrivate->runloop(library);

  return 0;
}
