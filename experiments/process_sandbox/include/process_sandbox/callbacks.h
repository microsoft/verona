// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include "sandbox.h"

#include <limits.h>
/**
 * This file contains the mechanism for servicing callbacks from the sandbox.
 */
namespace sandbox
{
  class Library;
  /**
   * The kind of callback.  This is used to dispatch the callback to the
   * correct handler.
   */
  enum CallbackKind
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
     * Total number of built-in callback kinds.
     */
    BuiltInCallbackKindCount,
    /**
     * User-defined callback numbers start here.  User for callbacks from the
     * sandbox into Verona code and will itself be multiplexed.
     */
    FirstUserFunction = BuiltInCallbackKindCount,
  };

  /**
   * The body of the callback request message.  This is sent over a UNIX domain
   * socket.  The privileged code has access to the sandbox's memory; however,
   * and so most of the message payload is in sandbox-owned memory within the
   * sandbox.  The payload sent over the socket is small enough that the kernel
   * can trivially buffer it and do two-copy I/O without any noticeable
   * overhead.  The host can then do one-copy I/O from the shared region to
   * avoid TOCTOU bugs, without needing to lock pages or any of the other
   * operations that make this expensive in an in-kernel implementation.
   */
  struct CallbackRequest
  {
    /**
     * The kind of message.
     */
    CallbackKind kind;
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
   * The response to a callback.
   */
  struct CallbackResponse
  {
    /**
     * The result value.  Currently, all callbacks follow the POSIX system call
     * model of returning a single integer.  Each response is optionally
     * accompanied by a file descriptor and most calls ignore this value if
     * they receive a file descriptor.
     */
    uintptr_t response;
  };

  /**
   * Base class for wrapping callbacks that handle callbacks.
   */
  struct CallbackHandlerBase
  {
    /**
     * The result type for any callback handler.  Note that callbacks are able
     * to cheaply allocate memory in the sandbox and so have a simple mechanism
     * for passing richer return values indirectly.
     */
    struct Result
    {
      /**
       * The return integer value.  This is passed directly via the socket.
       */
      intptr_t integer = -ENOSYS;

      /**
       * The file descriptor to return.  Optional.
       */
      platform::Handle handle;

      /**
       * Constructor, allows implicit creation from just an integer value.
       */
      Result(intptr_t i) : integer(i) {}

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
    virtual Result invoke(Library&, struct CallbackRequest)
    {
      return -ENOSYS;
    }

    /**
     * Virtual destructor.
     */
    virtual ~CallbackHandlerBase() {}
  };

  /**
   * Callback handler, wraps a callable object that actually handles the
   * message. The wrapper is responsible for defensively copying the argument
   * structure out so that the wrapped function does not have to worry about
   * truncated arguments and can ignore TOCTOU issues unless it follows pointers
   * in the argument frame.
   */
  template<typename T>
  class CallbackHandler : public CallbackHandlerBase
  {
    /**
     * The implementation of the handler.
     */
    std::function<CallbackHandlerBase::Result(Library&, T&)> fn;

    /**
     * Invoke the handler after checking that the arguments are of the correct
     * size and copying them out of the sandbox.
     */
    CallbackHandlerBase::Result
    invoke(Library& lib, struct CallbackRequest req) override
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
      memcpy(
        reinterpret_cast<void*>(arg.get()),
        reinterpret_cast<void*>(req.data),
        sizeof(T));
      return fn(lib, *arg);
    }

  public:
    /**
     * Templated constructor, allows this to be constructed from any function
     * (or other callable object) of the right signature.
     */
    template<typename X>
    CallbackHandler(X&& f) : fn(f)
    {}
  };

  /**
   * Helper that constructs a unique pointer to a callback handler of the right
   * type for the callable argument as a pointer to the base class.
   */
  template<typename T>
  std::unique_ptr<CallbackHandlerBase> make_callback_handler(
    std::function<CallbackHandlerBase::Result(Library&, T&)> fn)
  {
    return std::make_unique<CallbackHandler<T>>(fn);
  }

  /**
   * Type encapsulating the map from native type to the type that we use in the
   * RPC message.
   *
   * TODO: These currently assume the same ABI in and out of the sandbox.  This
   * may not always apply, for example if we run a 32-bit library on a 64-bit
   * host (or a 64-bit library on a CHERI host).
   *
   * This uses `uintptr_t` for pointers, to force an explicit cast before
   * access.
   *
   * Note: `uintptr_t` can't travel through pipes in CHERI systems, but
   * hopefully on a CHERI system we'd be using in-address-space
   * compartmentalisation rather than this mechanism.
   */
  template<typename T>
  struct RPCType
  {
    /**
     * The type that should be used to represent `T` in RPC messages.
     *
     * Currently, all types are passed as-is except pointers, which are
     * stored as `uintptr_t`.
     */
    using type = std::conditional_t<std::is_pointer_v<T>, uintptr_t, T>;
  };

  namespace internal
  {
    /**
     * Template that takes an array of types and produces a tuple of types that
     * contain the `RPCType` transform applied to each template argument.  For
     * example, `RPCTypes<char, void*>::types` will be `std::tuple<char,
     * uintptr_t>`.
     */
    template<typename T, typename... Ts>
    struct RPCTypes
    {
      /**
       * A tuple type representing the RPC type to use for a function that
       * takes all of the types provided to this class as template arguments.
       */
      using type = decltype(std::tuple_cat(
        std::declval<std::tuple<typename RPCType<T>::type>>(),
        std::declval<typename RPCTypes<Ts...>::type>()));
    };

    /**
     * Base case for the `RPCTypes`recursive template.
     */
    template<typename T>
    struct RPCTypes<T>
    {
      using type = std::tuple<typename RPCType<T>::type>;
    };

    /**
     * Base type for system call arguments.  This forward declaration is
     * required because we rely on partial specialisation for all valid
     * instantiations.
     */
    template<CallbackKind Kind, typename T>
    struct SyscallArgsBase;

    /**
     * Partial specialisation of `SyscallArgsBase`.  This is the real version:
     * It takes a system call kind and a function type as arguments and exposes
     * the inferred RPC type.  This should be used only by `SyscallArgs`.
     */
    template<CallbackKind Kind, typename R, typename... Args>
    struct SyscallArgsBase<Kind, R(Args...)>
    {
      /**
       * The callback kind.
       */
      static constexpr CallbackKind kind = Kind;

      /**
       * The type of the function that implements the system call.
       */
      using function_type = R(Args...);

      /**
       * The type of the RPC message.  This is a tuple type capable of carrying
       * the arguments from `function_type` through a pipe.
       */
      using rpc_type = typename RPCTypes<Args...>::type;
    };

    /**
     * On some platforms, syscalls are marked explicitly as `noexcept` (which
     * makes sense, because they don't throw exceptions).  The presence or
     * absence of this annotation doesn't affect this class, so match this by
     * delegating to the non-`noexcept` variant.
     */
    template<CallbackKind Kind, typename R, typename... Args>
    struct SyscallArgsBase<Kind, R(Args...) noexcept>
    : SyscallArgsBase<Kind, R(Args...)>
    {};
  }

  /**
   * System call arguments metadata structure.  This type is used as a map from
   * a `CallbackKind` to some type metadata and so only the concrete
   * specialisations should be used.  The generic version does not export
   * anything and so will cause compile failures if directly instantiated.
   */
  template<CallbackKind Kind>
  struct SyscallArgs
  {};

  SANDBOX_GCC_DIAGNOSTIC_IGNORE("-Wignored-attributes")

  /**
   * System call arguments for the `open` call.  Note that this cannot infer the
   * type from `open` because POSIX requires `open` to be declared as
   * `open(const char *, int, ...)`.  The variadic arguments are only ever a
   * `mode_t` but C does not provide default arguments and so this is the only
   * way of expressing an optional argument.
   */
  template<>
  struct SyscallArgs<Open>
  : internal::SyscallArgsBase<Open, int(const char*, int, mode_t)>
  {};

  /**
   * The system call arguments for the `access` call.  Inferred from the
   * declaration of the system call.
   */
  template<>
  struct SyscallArgs<Access>
  : internal::SyscallArgsBase<Access, decltype(::access)>
  {};

  /**
   * The system call arguments for the `stat` call.  Inferred from the
   * declaration of the system call.
   */
  template<>
  struct SyscallArgs<Stat> : internal::SyscallArgsBase<Stat, decltype(::stat)>
  {};

  SANDBOX_GCC_DIAGNOSTIC_POP()
}
