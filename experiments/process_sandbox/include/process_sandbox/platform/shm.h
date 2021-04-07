// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * Shared memory interface.  This file exposes a platform-specific class with
 * the following interface:
 *
 * ```
 * struct SharedMemoryMap
 * {
 *   SharedMemoryMap(uint8_t log2_size);
 *   Handle& get_handle()
 *   size_t get_size()
 *   void* get_base()
 *
 * };
 * ```
 *
 * The constructor takes the base-2 logarithm of the size, allocates an
 * anonymous shared memory object, and then maps it (naturally aligned)
 * somewhere in the address space.  The `get_handle` method returns the
 * platform-specific handle to the object, for passing to the child process.
 * The remaining two accessors get the size and base address of the mapping.
 */

#include "shm_posix.h"
namespace sandbox
{
  namespace platform
  {
    /*

    struct SharedMemoryMapping
    {
      Handle get_fd();
      SharedMemoryMapping(uint8_t log2_size);
    };
    */
    using SharedMemoryMap =
#if defined(__linux__)
      // On Linux, use the generic (unaligned) POSIX map and use MemFD if
      // it's available.
      SharedMemoryMapPOSIX<
#  ifdef USE_MEMFD
        detail::SharedMemoryObjectMemFD
#  else
        detail::SharedMemoryObjectPOSIX
#  endif
        >
    // On other *NIX platforms, use `MAP_ALIGNED` if it's available, fall
    // back to generic unaligned `mmap` if not.  If `SHM_ANON` is
    // available, use that, otherwise fall back to generic POSIX shared
    // memory.
#elif defined(__unix__)
#  ifdef MAP_ALIGNED
      SharedMemoryMapMMapAligned<
#  else
      SharedMemoryMapPOSIX<
#  endif
#  ifdef SHM_ANON
        detail::SharedMemoryObjectShmAnon
#  else
        detail::SharedMemoryObjectPOSIX
#  endif
        >
#else
#  error Anonymous shared memory not implemented for your platform
#endif
      ;
  }
}
