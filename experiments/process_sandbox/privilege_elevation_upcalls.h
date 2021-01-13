// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include "sandbox.hh"

#include <limits.h>
/**
 * This file contains the mechanism for servicing upcalls from the sandbox.
 */
namespace sandbox
{
  class SandboxedLibrary;
  /**
   * The kind of upcall.  This is used to dispatch the upcall to the correct
   * handler.
   */
  enum UpcallKind
  {
    /**
     * Proxying an `open` system call.
     */
    Open,
    /**
     * Proxying a `stat` system call.
     */
    Stat,
    /**
     * Proxying an `access` system call.
     */
    Access,
    /**
     * Proxying an `openat` system call.
     */
    OpenAt,
    /**
     * Total number of built-in upcall kinds.
     */
    BuiltInUpcallKindCount,
    /**
     * User-defined callback numbers start here.  User for upcalls from the
     * sandbox into Verona code and will itself be multiplexed.
     */
    FirstUserFunction = BuiltInUpcallKindCount,
  };

  /**
   * The body of the upcall request message.  This is sent over a UNIX domain
   * socket.  The privileged code has access to the sandbox's memory; however,
   * and so most of the message payload is in sandbox-owned memory within the
   * sandbox.  The payload sent over the socket is small enough that the kernel
   * can trivially buffer it and do two-copy I/O without any noticeable
   * overhead.  The host can then do one-copy I/O from the shared region to
   * avoid TOCTOU bugs, without needing to lock pages or any of the other
   * operations that make this expensive in an in-kernel implementation.
   */
  struct UpcallRequest
  {
    /**
     * The kind of message.
     */
    UpcallKind kind;
    /**
     * The size of the message payload (`data`).  All messages are fixed size,
     * and so it should be possible to deduce this from the `kind`, but this
     * acts as an extra sanity check.
     */
    size_t size;
    /**
     * The payload of this message.  This is a pointer into the sandbox's heap.
     * It is up to the receiver of this to validate this value.  In a CHERI
     * coprocess world, this would require rederivation and so introduce
     * confused deputy attacks that could bypass in-sandbox memory safety.
     * That should be fixed by replacing the use of UNIX domain sockets with a
     * cocall or similar abstraction.
     */
    uintptr_t data;
  };

  /**
   * The response to an upcall.
   */
  struct UpcallResponse
  {
    /**
     * The result value.  Currently, all upcalls follow the POSIX system call
     * model of returning a single integer.  Each response is optionally
     * accompanied by a file descriptor and most calls ignore this value if
     * they receive a file descriptor.
     */
    uintptr_t response;
  };

  /**
   * Base class for wrapping callbacks that handle upcalls.
   */
  struct UpcallHandlerBase
  {
    /**
     * The result type for any upcall handler.  Note that upcalls are able to
     * cheaply allocate memory in the sandbox and so have a simple mechanism
     * for passing richer return values indirectly.
     */
    struct Result
    {
      /**
       * The return integer value.  This is passed directly via the socket.
       */
      uintptr_t integer = -ENOSYS;

      /**
       * The file descriptor to return.  Optional.
       */
      platform::Handle handle;

      /**
       * Constructor, allows implicit creation from just an integer value.
       */
      Result(uintptr_t i) : integer(i) {}

      /**
       * Constructor, allows implicit creation from just a handle.
       */
      Result(platform::Handle&& h) : handle(std::move(h)) {}

      /**
       * Default constructor.
       */
      Result() = default;
    };

    /**
     * Invoke the handler.  This provides a type-safe interface that wraps the
     * templated version.
     */
    virtual Result invoke(SandboxedLibrary&, struct UpcallRequest)
    {
      return -ENOSYS;
    }

    /**
     * Virtual destructor.
     */
    virtual ~UpcallHandlerBase() {}
  };

  /**
   * Upcall handler, wraps a callable object that actually handles the message.
   * The wrapper is responsible for defensively copying the argument structure
   * out so that the wrapped function does not have to worry about truncated
   * arguments and can ignore TOCTOU issues unless it follows pointers in the
   * argument frame.
   */
  template<typename T>
  class UpcallHandler : public UpcallHandlerBase
  {
    /**
     * The implementation of the handler.
     */
    std::function<UpcallHandlerBase::Result(SandboxedLibrary&, T&)> fn;

    /**
     * Invoke the handler after checking that the arguments are of the correct
     * size and copying them out of the sandbox.
     */
    UpcallHandlerBase::Result
    invoke(SandboxedLibrary& lib, struct UpcallRequest req) override
    {
      if (req.size != sizeof(T))
      {
        return Result();
      }
      if (!lib.contains(reinterpret_cast<void*>(req.data), req.size))
      {
        return Result();
      }
      // Defensively copy the request body out of the sandbox.
      auto arg = std::make_unique<T>();
      memcpy(arg.get(), reinterpret_cast<void*>(req.data), sizeof(T));
      return fn(lib, *arg);
    }

  public:
    /**
     * Templated constructor, allows this to be constructed from any function
     * (or other callable object) of the right signature.
     */
    template<typename X>
    UpcallHandler(X&& f) : fn(f)
    {}
  };

  /**
   * Helper that constructs a unique pointer to an Upcall handler of the right
   * type for the callable argument as a pointer to the base class.
   */
  template<typename T>
  std::unique_ptr<UpcallHandlerBase> make_upcall_handler(
    std::function<UpcallHandlerBase::Result(SandboxedLibrary&, T&)> fn)
  {
    return std::make_unique<UpcallHandler<T>>(fn);
  }

  /**
   * Upcall argument structures.
   *
   * TODO: These currently assume the same ABI in and out of the sandbox.  This
   * may not always apply, for example if we run a 32-bit library on a 64-bit
   * host (or a 64-bit library on a CHERI host).
   *
   * This uses `uintptr_t` for pointers, to force an explicit cast before
   * access.
   */
  namespace UpcallArgs
  {
    /**
     * Arguments for an open upcall.
     */
    struct Open
    {
      /**
       * The path to open.  This must be copied out of the sandbox before use.
       */
      uintptr_t path;
      /**
       * The open flags.
       */
      int flags;
      /**
       * The mode.
       */
      mode_t mode;
    };

    /**
     * Arguments to the `stat` system call.
     */
    struct Stat
    {
      /**
       * The path.  This must be copied out of the sandbox before use.
       */
      uintptr_t path;
      /**
       * The location of the `stat` buffer.  This can be passed directly to the
       * system call after checking that it is completely within the sandbox.
       */
      uintptr_t statbuf;
    };
  }

}
