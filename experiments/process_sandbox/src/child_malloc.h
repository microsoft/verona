// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include <aal/aal.h>
#include <ds/bits.h>
#include <pal/pal.h>
#include <stdint.h>

namespace snmalloc
{
  template<typename T>
  struct SuperslabMap;
  class Superslab;
  class Mediumslab;
  class Largeslab;
}

namespace sandbox
{
  /**
   * The proxy pagemap.  In the sandboxed process, there are two parts of the
   * pagemap.  The child process maintains a private pagemap for process-local
   * memory, the parent process maintains a fragment of this pagemap
   * corresponding to the shared memory region.
   *
   * This class is responsible for managing the composition of the two page
   * maps.  All queries are local but updates must be forwarded to either the
   * parent process or the private pagemap.
   */
  struct ProxyPageMap
  {
    /**
     * Singleton instance of this class.
     */
    static ProxyPageMap p;
    /**
     * Accessor, returns the singleton instance of this class.
     */
    static ProxyPageMap& pagemap()
    {
      return p;
    }
    /**
     * Helper function used by the set methods that are part of the page map
     * interface.  Requires that the address (`p`) is in the shared region.
     * This writes the required change as a message to the parent process and
     * spins waiting for the parent to make the required update.
     */
    void set(snmalloc::address_t p, uint8_t x);
    /**
     * Get the pagemap entry for a specific address.
     */
    static uint8_t get(snmalloc::address_t p);
    /**
     * Get the pagemap entry for a specific address.
     */
    static uint8_t get(void* p);
    /**
     * Type-safe interface for setting that a particular memory region contains
     * a superslab. This calls `set`.
     */
    void set_slab(snmalloc::Superslab* slab);
    /**
     * Type-safe interface for notifying that a region no longer contains a
     * superslab.  Calls `set`.
     */
    void clear_slab(snmalloc::Superslab* slab);
    /**
     * Type-safe interface for notifying that a region no longer contains a
     * medium slab.  Calls `set`.
     */
    void clear_slab(snmalloc::Mediumslab* slab);
    /**
     * Type-safe interface for setting that a particular memory region contains
     * a medium slab. This calls `set`.
     */
    void set_slab(snmalloc::Mediumslab* slab);
    /**
     * Type-safe interface for setting the pagemap values for a region to
     * indicate a large allocation.
     */
    void set_large_size(void* p, size_t size);
    /**
     * The inverse operation of `set_large_size`, updates a range to indicate
     * that it is not in use.
     */
    void clear_large_size(void* p, size_t size);
  };

  /**
   * The proxy memory provider.  This uses a simple RPC protocol to forward all
   * requests to the parent process, which validates the arguments and forwards
   * them to the trusted address-space manager for the sandbox.
   */
  struct MemoryProviderProxy
  {
    /**
     * The PAL that we use inside the sandbox.  This is incapable of
     * allocating memory.
     */
    typedef snmalloc::PALNoAlloc<snmalloc::DefaultPal> Pal;

    /**
     * Pop a large allocation from the stack to the address space manager in
     * the parent process corresponding to a large size class.
     */
    void* pop_large_stack(size_t large_class);

    /**
     * Push a large allocation to the address space manager in the parent
     * process.
     */
    void push_large_stack(snmalloc::Largeslab* slab, size_t large_class);

    /**
     * Reserve committed memory of a large size class size by calling into the
     * parent to request address space in the shared region.
     */
    void* reserve_committed(size_t large_class) noexcept;

    /**
     * Reserve a range of memory identified by a size, not a size class.  This
     * is used only in `alloc_chunk` and is not inlined because its
     * implementation needs to refer to snmalloc size classes, which are
     * defined only in a header inclusion that requires this class to be
     * defined.
     */
    void* reserve_committed_size(size_t size) noexcept;

    /**
     * Public interface to reserve memory.  Ignores the `committed` argument,
     * the shared memory is always committed.
     */
    template<bool committed>
    void* reserve(size_t large_class) noexcept
    {
      return reserve_committed(large_class);
    }

    /**
     * Factory method, used by the `Singleton` helper.  This is responsible for
     * any bootstrapping needed to communicate with the parent.
     */
    static MemoryProviderProxy* make() noexcept;

    /**
     * Allocate a chunk.  This implementation is wasteful, rounding the
     * requested sizes up to the smallest large size class (typically one MiB).
     * This is called only twice in normal use for a sandbox, so wasting a bit
     * of address space is not the highest priority fix yet.  Given how little
     * this is actually needed, a better implementation would reserve some
     * space in the non-heap shared memory region for these requests.
     */
    template<typename T, size_t alignment, typename... Args>
    T* alloc_chunk(Args&&... args)
    {
      // Cache line align
      size_t size = snmalloc::bits::align_up(sizeof(T), 64);
      size = snmalloc::bits::next_pow2(snmalloc::bits::max(size, alignment));
      void* p = reserve_committed_size(size);
      if (p == nullptr)
        return nullptr;

      return new (p) T(std::forward<Args...>(args)...);
    }
  };
}

#define SNMALLOC_DEFAULT_CHUNKMAP sandbox::ProxyPageMap
#define SNMALLOC_DEFAULT_MEMORY_PROVIDER sandbox::MemoryProviderProxy
#ifdef __FreeBSD__
#  define SNMALLOC_USE_THREAD_CLEANUP 1
#endif
#include "override/malloc.cc"
