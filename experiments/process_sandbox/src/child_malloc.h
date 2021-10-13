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
#include <snmalloc_core.h>

namespace snmalloc
{
  class Superslab;
  class Mediumslab;
  class Largeslab;
}

namespace sandbox
{
  /**
   * The snmalloc configuration used for the child process.
   */
  class SnmallocGlobals : public snmalloc::CommonConfig
  {
    /**
     * Non-templated version of reserve_with_left_over.  Calls the parent
     * process to request a new chunk of memory.
     */
    static snmalloc::capptr::Chunk<void> reserve(size_t size);

    /**
     * Private address-space manager.  Used to manage allocations that are
     * not shared with the parent.
     */
    inline static snmalloc::AddressSpaceManager<snmalloc::DefaultPal>
      private_asm;

  public:
    /**
     * Interface to a pagemap that is private to the child.  This is used with
     * the private address-space manager so that the child process can track
     * metadata about unused memory without needing an RPC to the parent.
     */
    struct PrivatePagemap
    {
      /**
       * Empty local state.  This is required to exist by snmalloc.
       */
      struct LocalState
      {};

      /**
       * Private pagemap, allocated in memory owned exclusively by the child.
       *
       * This spans the entire child's address space. Any parts of it
       * that cover the range shared with the parent will be unused.
       */
      inline static snmalloc::FlatPagemap<
        snmalloc::MIN_CHUNK_BITS,
        snmalloc::MetaEntry,
        snmalloc::DefaultPal,
        /*fixed range*/ false>
        pagemap;

      /**
       * Return the metadata associated with an address.  This reads the
       * private pagemap.
       */
      template<bool potentially_out_of_range = false>
      SNMALLOC_FAST_PATH static const snmalloc::MetaEntry&
      get_metaentry(LocalState*, snmalloc::address_t p)
      {
        return pagemap.template get<potentially_out_of_range>(p);
      }

      /**
       * Set the metadata associated with an address range.  Updates the private
       * pagemap directly.
       */
      SNMALLOC_FAST_PATH
      static void set_metaentry(
        LocalState*, snmalloc::address_t p, size_t size, snmalloc::MetaEntry t)
      {
        for (auto a = p; a < p + size; a += snmalloc::MIN_CHUNK_SIZE)
        {
          pagemap.set(a, t);
        }
      }

      /**
       * Ensure that the range is valid in the private pagemap.
       */
      static void
      register_range(LocalState*, snmalloc::address_t p, size_t size)
      {
        pagemap.register_range(p, size);
      }
    };

    /**
     * Expose a PAL that doesn't do allocation.
     */
    using Pal = snmalloc::PALNoAlloc<snmalloc::DefaultPal>;

    /**
     * Thread-local state.  Currently not used.
     */
    struct LocalState
    {};

    /**
     * Adaptor for the pagemap that is managed by the parent.  This is backed
     * by a shared-memory object that is passed into the child on process start
     * and is then mapped read-only in the child.  All updates require an RPC
     * to the parent, which will validate the updates and install them.
     */
    struct Pagemap
    {
      /**
       * Local state for pagemap updates.  Currently unused.
       */
      using LocalState = SnmallocGlobals::LocalState;

      /**
       * The pagemap that spans the entire address space.  This uses a read-only
       * mapping of a shared memory region as its backing store.
       */
      inline static snmalloc::FlatPagemap<
        snmalloc::MIN_CHUNK_BITS,
        snmalloc::MetaEntry,
        snmalloc::DefaultPal,
        /*fixed range*/ false>
        pagemap;

      /**
       * Return the metadata associated with an address.  This reads the
       * read-only mapping of the pagemap directly.
       */
      template<bool potentially_out_of_range = false>
      SNMALLOC_FAST_PATH static const snmalloc::MetaEntry&
      get_metaentry(LocalState*, snmalloc::address_t p)
      {
        return pagemap.template get<potentially_out_of_range>(p);
      }

      /**
       * Set the metadata associated with an address range.  Sends an RPC to the
       * parent, which validates and inserts the entry.
       */
      SNMALLOC_FAST_PATH
      static void set_metaentry(
        LocalState*, snmalloc::address_t p, size_t size, snmalloc::MetaEntry t);

      /**
       * Ensure that the range is valid.  This is a no-op: the parent is
       * responsible for ensuring that the pagemap covers the entire address
       * range.
       */
      static void register_range(LocalState*, snmalloc::address_t, size_t) {}
    };

    /**
     * Allocate a chunk of memory and install its metadata in the pagemap.
     * This performs a single RPC that validates the metadata and then
     * allocates and installs the entry.
     */
    static std::pair<snmalloc::capptr::Chunk<void>, snmalloc::Metaslab*>
    alloc_chunk(
      LocalState* local_state,
      size_t size,
      snmalloc::RemoteAllocator* remote,
      snmalloc::sizeclass_t sizeclass);

    /**
     * Allocate metadata.  This allocates non-shared memory for metaslabs and
     * shared memory for allocators.
     */
    template<typename T>
    static snmalloc::capptr::Chunk<void>
    alloc_meta_data(LocalState*, size_t size);

    /**
     * The allocator pool type used to allocate per-thread allocators.
     */
    using AllocPool =
      snmalloc::PoolState<snmalloc::CoreAllocator<SnmallocGlobals>>;

    /**
     * The state associated with the chunk allocator.
     */
    inline static snmalloc::ChunkAllocatorState chunk_alloc_state;

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

    /**
     * Returns the singleton instance of the chunk allocator state.
     */
    static snmalloc::ChunkAllocatorState& get_slab_allocator_state(void*)
    {
      return chunk_alloc_state;
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

#include <override/malloc.cc>
