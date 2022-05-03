// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include <stdint.h>

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
     * Allocate a chunk.  The arguments are:
     *  - The size to allocate
     *  - The address of the metadata.
     *  - The address of the message queue and the sizeclass
     *
     * Note that the address of the metadata should never be
     * accessed from outside the sandbox and so does not need to be in the
     * shared memory region.
     */
    AllocChunk,

    /**
     * Deallocate a chunk.  The arguments are:
     *
     * - The address of the chunk.
     * - The size of the chunk.
     */
    DeallocChunk,
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
     * The arguments.  The interpretation of this is depends on the call.
     */
    uintptr_t args[4];
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
