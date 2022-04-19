// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#ifdef __unix__
#  include <pthread.h>
#endif

#include <snmalloc/snmalloc_core.h>

namespace sandbox
{
  /**
   * Class representing a view of a shared memory region.  This provides both
   * the parent and child views of the region.
   */
  struct SharedMemoryRegion
  {
    /**
     * The start of the sandbox region.  Note: This is writeable from within
     * the sandbox and should not be trusted outside.
     */
    void* start;

    /**
     * The end of the sandbox region.  Note: This is writeable from within
     * the sandbox and should not be trusted outside.
     */
    void* end;

    /**
     * A flag indicating that the parent has instructed the sandbox to exit.
     */
    std::atomic<bool> should_exit = false;
    /**
     * The index of the function currently being called.  This interface is not
     * currently reentrant.
     */
    int function_index;
    /**
     * A pointer to the tuple (in the shared memory range) that contains the
     * argument frame provided by the sandbox caller.
     */
    void* msg_buffer = nullptr;
    /**
     * The message queue for the parent's allocator.  This is stored in the
     * shared region because the child must be able to free memory allocated by
     * the parent.
     */
    snmalloc::RemoteAllocator allocator_state;

    /**
     * A token that is logically passed from the parent to the child and back
     * again, where each hands control to the other.
     */
    struct
    {
      /**
       * Semaphore that the child sleeps on when it's not running.
       */
      platform::OneBitSem child{0};
      /**
       * Semaphore that the parent sleeps on when the child is running.
       */
      platform::OneBitSem parent{0};
      /**
       * Flag indicating whether the child is executing.  Used only for
       * debugging.
       */
      std::atomic<bool> is_child_executing = false;
      /**
       * Monotonic flag indicating that the child has finished loading.  This
       * is checked during the first call into a sandbox to process any pending
       * callbacks before the first invocation.
       */
      std::atomic<bool> is_child_loaded = false;
      /**
       * The current depth of callbacks.
       *
       * FIXME: Rentrancy with multiple threads in the child is not working.
       */
      std::atomic<int> callback_depth = 0;
    } token;

    /**
     * Tear down the parent-owned contents of this shared memory region.
     */
    void destroy() {}
  };
}
