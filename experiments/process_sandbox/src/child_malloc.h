// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#ifdef __FreeBSD__
#  define SNMALLOC_USE_THREAD_CLEANUP 1
#endif
#define SNMALLOC_PLATFORM_HAS_GETENTROPY 1
#define SNMALLOC_PROVIDE_OWN_CONFIG 1
#include <process_sandbox/helpers.h>
#include <process_sandbox/sandbox_fd_numbers.h>
#include <process_sandbox/sandbox_meta_entry.h>
#include <snmalloc/snmalloc_core.h>

namespace snmalloc
{
  class Superslab;
  class Mediumslab;
  class Largeslab;
}

namespace sandbox
{
  /**
   * Helper range that deallocates memory provided by the PAL.
   */
  struct PalRange : public snmalloc::PalRange<snmalloc::DefaultPal>
  {
    /**
     * This range is intended to be used directly by the small buddy allocator
     * without a pagemap to store extra metadata and so must support aligned
     * allocation, even if the underlying PAL does not.
     */
    static constexpr bool Aligned = true;

    /**
     * All operations on the address space are locked by the kernel.
     */
    static constexpr bool ConcurrencySafe = true;

    /**
     * Deallocate memory using munmap.
     */
    void dealloc_range(snmalloc::capptr::Arena<void> base, size_t size);

    /**
     * Allocate memory, trimming to guarantee alignment if necessary.
     */
    snmalloc::capptr::Arena<void> alloc_range(size_t size)
    {
      if constexpr (snmalloc::pal_supports<
                      snmalloc::AlignedAllocation,
                      snmalloc::DefaultPal>)
      {
        return snmalloc::PalRange<snmalloc::DefaultPal>::alloc_range(size);
      }
      else
      {
        size_t overallocation = size * 2;
        auto alloc =
          snmalloc::PalRange<snmalloc::DefaultPal>::alloc_range(overallocation);
        auto end = snmalloc::pointer_offset(alloc, size * 2);
        auto aligned_base = snmalloc::pointer_align_up(alloc, size);
        auto aligned_end = snmalloc::pointer_offset(aligned_base, size);
        if (end.unsafe_ptr() > aligned_end.unsafe_ptr())
        {
          dealloc_range(
            aligned_end, end.unsafe_uintptr() - aligned_end.unsafe_uintptr());
        }
        if (aligned_base.unsafe_ptr() > alloc.unsafe_ptr())
        {
          dealloc_range(
            alloc, aligned_base.unsafe_uintptr() - alloc.unsafe_uintptr());
        }
        return aligned_base;
      }
    }
  };

  /**
   * The snmalloc configuration used for the child process.
   */
  class SnmallocGlobals : public snmalloc::CommonConfig
  {
    /**
     * Private allocator.  Used to manage metadata allocations, which are
     * not shared with the parent.
     */
    inline static snmalloc::Pipe<PalRange, snmalloc::SmallBuddyRange>
      metadata_range;

  public:
    /**
     * Expose a PAL that doesn't do allocation.
     */
    using Pal = snmalloc::PALNoAlloc<snmalloc::DefaultPal>;

    using PagemapEntry = SandboxMetaEntry;

    /**
     * The pagemap that spans the entire address space.  This uses a read-only
     * mapping of a shared memory region as its backing store.
     */
    inline static snmalloc::FlatPagemap<
      snmalloc::MIN_CHUNK_BITS,
      PagemapEntry,
      snmalloc::DefaultPal,
      /*fixed range*/ false>
      pagemap;

    /**
     * Thread-local state.  Currently not used.
     */
    struct LocalState
    {};

    class Backend
    {
    public:
      /**
       * This back end does not need to hold any extra metadata and so exports
       * the default slab metadata type.
       */
      using SlabMetadata = snmalloc::FrontendSlabMetadata;

      /**
       * Allocate a chunk of memory and install its metadata in the pagemap.
       * This performs a single RPC that validates the metadata and then
       * allocates and installs the entry.
       */
      static std::pair<snmalloc::capptr::Chunk<void>, SlabMetadata*>
      alloc_chunk(LocalState& local_state, size_t size, uintptr_t ras);

      static void dealloc_chunk(
        LocalState& local_state,
        SlabMetadata& meta_common,
        snmalloc::capptr::Alloc<void> start,
        size_t size);

      /**
       * Allocate metadata.  This allocates non-shared memory for metaslabs and
       * shared memory for allocators.
       */
      template<typename T>
      static snmalloc::capptr::Arena<void>
      alloc_meta_data(LocalState*, size_t size);

      /**
       * Return the metadata associated with an address.  This reads the
       * read-only mapping of the pagemap directly.
       */
      template<bool potentially_out_of_range = false>
      static const auto& get_metaentry(snmalloc::address_t p)
      {
        return pagemap.template get<potentially_out_of_range>(p);
      }
    };

    /**
     * The allocator pool type used to allocate per-thread allocators.
     */
    using AllocPool =
      snmalloc::PoolState<snmalloc::CoreAllocator<SnmallocGlobals>>;

    /**
     * The concrete instance of the pool allocator.
     */
    inline static AllocPool alloc_pool;

  public:
    /**
     * Returns the allocation pool.
     */
    static AllocPool& pool()
    {
      return alloc_pool;
    }

    /**
     * Ensure that all of the early bootstrapping is done.
     */
    static void ensure_init() noexcept;

    /**
     * Returns true if the system has bootstrapped, false otherwise.
     */
    static bool is_initialised();

    /**
     * Message queues are currently always allocated inline for
     * in-sandbox allocators.  When we move to dynamically creating
     * shared memory objects one per chunk then they will move to a
     * separate place. For now, all options are the defaults.
     */
    constexpr static snmalloc::Flags Options{};

    /**
     * Register per-thread cleanup.
     */
    static void register_clean_up()
    {
#ifndef SNMALLOC_USE_THREAD_CLEANUP
      snmalloc::register_clean_up();
#endif
    }
  };
}

namespace snmalloc
{
  /**
   * The standard allocator type that we provide.
   */
  using Alloc = LocalAllocator<sandbox::SnmallocGlobals>;
}

#include <snmalloc/override/malloc.cc>
