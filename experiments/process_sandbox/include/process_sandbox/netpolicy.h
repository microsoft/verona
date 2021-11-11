// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This file contains the abstractions required for granting network access in
 * a sandbox.  Sandboxes are not permitted to do anything directly that touches
 * a global namespace and so may not connect or bind sockets.  Any attempt to
 * use the network must be proxied.
 */

#pragma once
#include <netdb.h>
#include <optional>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace sandbox
{
  /**
   * Class encapsulating a network access policy for a sandbox.  This allows
   * specific access to some network resources to be granted to a sandbox.
   */
  class NetworkPolicy
  {
    /**
     * C++17 made noexcept part of the type system, so `R(Args...) noexcept` is
     * not the same as `R(Args...)`.  Unfortunately, `std::function` remains
     * specialised only for `R(Args...)` and so you cannot create a
     * `std::function<R(Args...) noexcept>`.  Fortunately, the `noexcept`
     * version is a subtype of the potentially throwing version (throwing 0
     * exceptions is a stronger constraint than throwing 0 or more exceptions)
     * and so we can just strip `noexcept` from the types.
     *
     * This helper provides a mechanism for stripping the `noexcept` qualifier
     * from a function type.
     */
    template<typename Res, typename... ArgTypes>
    struct remove_noexcept
    {};

    /**
     * Specialisation for types that are already not-`noexcept`, just forwards
     * the type.
     */
    template<typename Res, typename... ArgTypes>
    struct remove_noexcept<Res(ArgTypes...)>
    {
      /**
       * The type (already) without `noexcept`.
       */
      using type = Res(ArgTypes...);
    };

    /**
     * Specialisation for types that are `noexcept`, forwards the type without
     * `noexcept`.
     */
    template<typename Res, typename... ArgTypes>
    struct remove_noexcept<Res(ArgTypes...) noexcept>
    {
      /**
       * The type without `noexcept`.
       */
      using type = Res(ArgTypes...);
    };

    /**
     * Helper for avoiding the need to write `::type` after every use of
     * `remove_noexcept`.
     */
    template<typename Res, typename... ArgTypes>
    using remove_noexcept_t = typename remove_noexcept<Res, ArgTypes...>::type;

  public:
    /**
     * Enumeration describing the set of network operations that may be
     * controlled by this policy.
     */
    enum class NetOperation
    {
      /**
       * The `bind` function.  Assigns a local address to the socket, so
       * requires authorisation to access the global namespace.
       */
      Bind,
      /**
       * The `getaddrinfo` function.  Theoretically this does not require
       * access to the global namespace but in implementation it usually
       * requires the ability ot query DNS or similar.
       */
      GetAddrInfo,
      /**
       * Connect a socket to a remote address.  Requires access to the global
       * namespace to be authorised to connect to the specific remote address.
       */
      Connect
    };

  private:
    /**
     * Enumeration describing a simple deny-all or allow-all policy.
     */
    enum SimplePolicy
    {
      /**
       * Deny this operation.  This is 0 so that default initialisation gives a
       * default-deny policy.
       */
      Deny = 0,
      /**
       * Permit this operation, forwarding directly to the system
       * implementation.
       */
      Allow
    };

    /**
     * A policy.  This is either allow-all, deny-all, or invoke a callback to
     * handle each specific case.  The arguments to these functions are copied
     * before the callback is invoked, so the callback does not have to worry
     * about TOCTOU bugs.
     */
    template<typename Callback>
    using Policy =
      std::variant<SimplePolicy, std::function<remove_noexcept_t<Callback>>>;

    /**
     * The set of default implementations for each of the functions.
     */
    constexpr static std::tuple systemImplementations = {
      &bind, &getaddrinfo, &connect};

    /**
     * Helper to map from the `NetOperation` enumeration to the type of the
     * function required to implement it.
     */
    template<NetOperation Op>
    using NetOpFnType =
      typename std::remove_pointer_t<std::remove_reference_t<decltype(
        std::get<static_cast<int>(Op)>(systemImplementations))>>;

    /**
     * Helper that defines a `std::function` type matching one of the network
     * operations that can access the global namespace.
     */
    template<NetOperation Op>
    using NetOpCallbackType = std::function<remove_noexcept_t<NetOpFnType<Op>>>;

    /**
     * Policies for various network operations, indexed by `NetOperation`.
     */
    std::tuple<
      Policy<NetOpFnType<NetOperation::Bind>>,
      Policy<NetOpFnType<NetOperation::GetAddrInfo>>,
      Policy<NetOpFnType<NetOperation::Connect>>>
      policies;

  public:
    /**
     * Function used to free the return from any explicit `getaddrinfo`
     * handler.  This must be set if the handler returns anything other than
     * the result from the system's `getaddrinfo`.
     */
    std::function<remove_noexcept_t<decltype(::freeaddrinfo)>> freeaddrinfo{
      ::freeaddrinfo};

    /**
     * Get the policy for a given operation.
     *
     * Note: The return type uses `NetOpFnType` explicitly rather than `auto`
     * so that if the order of `NetOperation` and `policies` are out of sync we
     * will get a compile error.
     */
    template<NetOperation Op>
    Policy<NetOpFnType<Op>>& getPolicy()
    {
      return std::get<static_cast<int>(Op)>(policies);
    }

    /**
     * Deny the sandbox the ability to perform the operation specified by `Op`.
     */
    template<NetOperation Op>
    void deny()
    {
      getPolicy<Op>() = Deny;
    }

    /**
     * Allow the sandbox the ability to perform the operation specified by `Op`
     * with no additional checks.
     */
    template<NetOperation Op>
    void allow()
    {
      getPolicy<Op>() = Allow;
    }

    /**
     * Register a handler for the operation specified by `Op`.  This must have
     * the same signature as the underlying system function and may be used to
     * enforce arbitrary policy decisions.
     */
    template<NetOperation Op>
    void register_handler(NetOpCallbackType<Op> callback)
    {
      getPolicy<Op>() = callback;
    }

    /**
     * Invoke the socket API call with the given arguments, enforcing the
     * registered policy.  This does *not* check that the arguments are valid,
     * the caller is responsible for any checks and must copy arguments.
     */
    template<NetOperation Op, typename... Args>
    int invoke(Args... args)
    {
      auto& policy = getPolicy<Op>();
      return std::visit(
        [&](auto& v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, SimplePolicy>)
          {
            if (v == Deny)
            {
              errno = ENOSYS;
              return -1;
            }
            return std::get<static_cast<int>(Op)>(systemImplementations)(
              std::forward<Args>(args)...);
          }
          else
          {
            return v(std::forward<Args>(args)...);
          }
        },
        policy);
    }
  };
}
