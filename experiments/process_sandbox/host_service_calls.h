// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
/**
 * This file includes all of the types used for forwarding memory-provider
 * requests and pagemap updates from the sandbox to the parent.  This is a
 * trivial lightweight request-response protocol, using fixed-size binary
 * messages.
 *
 * This protocol is trivial and so is done as something very simple and ad-hoc.
 * It should be replaced by something more robust if it ever grows more
 * complex. After sandboxes are made reentrant, the same upcall mechanism used
 * for calling back into Verona can be used to replace this.
 */

namespace sandbox
{
  /**
   * The ID of the method that's being proxied.
   */
  enum HostServiceCallID : uintptr_t
  {
    /**
     * Push a large allocation to the stack.  The first argument is the address
     * of the slab, the second the large sizeclass.  The return value is unused.
     *
     * The slab must be within the shared memory region for the sandbox and so
     * must have been previously returned with a call to either
     * `MemoryProviderPopLargeStack` or `MemoryProviderReserve`.
     */
    MemoryProviderPushLargeStack,
    /**
     * Pop a large allocation from the stack.  The first argument is the large
     * sizeclass, the second is unused.  The return value is the address of the
     * large allocation that was popped from the stack, 0 indicates that the
     * stack was empty.
     */
    MemoryProviderPopLargeStack,
    /**
     * Reserve memory.  The first argument is the large sizeclass, the second
     * is unused.  The return value is the start of the reserved address range.
     *
     * Note: The `committed` template parameter is ignored, the shared memory
     * region is assumed to always be committed.
     */
    MemoryProviderReserve,
    /**
     * Set a chunk map element. The first argument is the address, the second
     * is the chunkmap entry.  The return value is unused.
     */
    ChunkMapSet,
    /**
     * Set a range in the chunk map.  The first argument is the base address,
     * the second is the base-2 logarithm of the size (rounded up).
     */
    ChunkMapSetRange,
    /**
     * Clear a range in the chunk map.  The first argument is the base address,
     * the second is the base-2 logarithm of the size (rounded up).
     */
    ChunkMapClearRange,
  };

  /**
   * The request structure.  Each call sends an instance of this structure
   * over the pipe to the parent.
   */
  struct HostServiceRequest
  {
    /**
     * The message ID for this message.  This is effectively a vtable index.
     */
    HostServiceCallID kind;

    /**
     * The first argument.  The interpretation of this is depends on the call.
     */
    uintptr_t arg0;
    /**
     * The second argument.  The interpretation of this is depends on the call.
     */
    uintptr_t arg1;
  };

  /**
   * The response to a privileged call.  This does not include a message ID
   * because all messages are processed in order and so the responses will
   * always be in the same order that the requests are sent.
   */
  struct HostServiceResponse
  {
    /**
     * Error flag.  0 indicates success, any other value indicates error.  In
     * the future, this may be extended to provide more meaningful information.
     */
    uintptr_t error;

    /**
     * The return value.
     */
    uintptr_t ret;
  };

}
