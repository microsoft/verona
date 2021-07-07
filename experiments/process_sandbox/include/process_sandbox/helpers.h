// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include <memory>

/**
 * Helper macro to apply a pragma in a macro, concatenating arguments as a
 * single string.
 */
#define SANDBOX_DO_PRAGMA(x) _Pragma(#x)

#ifdef __clang__
/**
 * Macro to silence a clang warning until the next
 * `SANDBOX_CLANG_DIAGNOSTIC_POP()`.  The argument is the name of the warning
 * as a string.  For example, if the flag to enable the warning is
 * `-Wwarning-with-false-positives` then the argument to this should be
 * `"-Wwarning-with-false-positives"`.
 *
 * When compiling with a non-clang compiler, this macro does nothing.
 */
#  define SANDBOX_CLANG_DIAGNOSTIC_IGNORE(x) \
    _Pragma("clang diagnostic push") \
      SANDBOX_DO_PRAGMA(clang diagnostic ignored x)
/**
 * Restores the set of enabled clang warnings to the set before the most recent
 * `SANDBOX_CLANG_DIAGNOSTIC_IGNORE()`.
 *
 * When compiling with a non-clang compiler, this macro does nothing.
 */
#  define SANDBOX_CLANG_DIAGNOSTIC_POP() _Pragma("clang diagnostic pop")
#else
#  define SANDBOX_CLANG_DIAGNOSTIC_IGNORE(x)
#  define SANDBOX_CLANG_DIAGNOSTIC_POP()
#endif

#if defined(__GNUC__) && !defined(__clang__)
/**
 * Macro to silence a gcc warning until the next
 * `SANDBOX_GCC_DIAGNOSTIC_POP()`.  The argument is the name of the warning
 * as a string.  For example, if the flag to enable the warning is
 * `-Wwarning-with-false-positives` then the argument to this should be
 * `"-Wwarning-with-false-positives"`.
 *
 * When compiling with a non-gcc compiler, this macro does nothing.
 */
#  define SANDBOX_GCC_DIAGNOSTIC_IGNORE(x) \
    _Pragma("GCC diagnostic push") SANDBOX_DO_PRAGMA(GCC diagnostic ignored x)
/**
 * Restores the set of enabled gcc warnings to the set before the most recent
 * `SANDBOX_GCC_DIAGNOSTIC_IGNORE()`.
 *
 * When compiling with a non-gcc compiler, this macro does nothing.
 */
#  define SANDBOX_GCC_DIAGNOSTIC_POP() _Pragma("GCC diagnostic pop")
#else
#  define SANDBOX_GCC_DIAGNOSTIC_IGNORE(x)
#  define SANDBOX_GCC_DIAGNOSTIC_POP()
#endif

#if __has_include(<experimental/source_location>)
#  include <experimental/source_location>
namespace sandbox
{
  using source_location = std::experimental::source_location;
}
#elif __has_include(<source_location>)
#  include <source_location>
namespace sandbox
{
  using source_location = std::source_location;
}
#else
// If we don't have a vaguely recent standard library then we don't get useful
// source locations.
namespace sandbox
{
  struct source_location
  {
    std::uint_least32_t line()
    {
      return 0;
    }
    const char* file_name()
    {
      return "{unknown file}";
    }
    const char* function_name()
    {
      return "{unknown function}";
    }
    static source_location current()
    {
      return {};
    }
  };
}
#endif
#include <fmt/format.h>
#include <pal/pal.h>

namespace sandbox
{
  /**
   * Handler for invariant failures.  Not inlined, this is always a slow
   * path.  This should be called only by `invariant`.
   */
  template<typename Msg>
  __attribute__((noinline)) void
  invariant_fail(Msg msg, fmt::format_args args, source_location sl)
  {
    std::string user_msg = fmt::vformat(msg, args);
    std::string final_msg = fmt::format(
      "{}:{} in {}: {}\n",
      sl.file_name(),
      sl.line(),
      sl.function_name(),
      user_msg);
    snmalloc::Pal::error(final_msg.c_str());
  }

  enum DebugOption
  {
    DebugOnly,
    DebugAndRelease,
    ReleaseOnly
  };

  /**
   * Invariant.  If `cond` is false, prints the message defined by `msg` and
   * the (optional) format-string arguments and aborts.  The caller should
   * never specify the `sl` parameter.
   *
   * Implementation detail: This receives the source location via the default
   * parameter and so has to put that after everything else, but C++ does not
   * allow arguments from a parameter pack in the middle of the argument list
   * and so we need to pack those into a tuple.
   */
  template<DebugOption Enable = DebugAndRelease, typename Msg, typename... Args>
  __attribute__((always_inline)) void invariant(
    bool cond,
    Msg msg = "Assertion failure",
    std::tuple<Args...> fmt_args = {},
    source_location sl = source_location::current())
  {
    constexpr bool isRelease =
#ifdef NDEBUG
      true
#else
      false
#endif
      ;

    constexpr bool isEnabled = ((Enable == DebugOnly) && !isRelease) ||
      ((Enable == ReleaseOnly) && isRelease) || (Enable == DebugAndRelease);
    if constexpr (isEnabled)
    {
      if (!cond)
      {
#if FMT_VERSION >= 70000
        using Char = fmt::char_t<Msg>;
        invariant_fail(
          msg,
          std::apply<fmt::format_arg_store<
            fmt::buffer_context<Char>,
            fmt::remove_reference_t<Args>...>(
            const Msg&, const fmt::remove_reference_t<Args>&...)>(
            fmt::make_args_checked<Args...>,
            std::tuple_cat(std::make_tuple(msg), fmt_args)),
          sl);
#else
        invariant_fail(
          msg,
          std::apply(
            fmt::make_format_args<fmt::format_context, Args...>, fmt_args),
          sl);
#endif
      }
    }
  }

  /**
   * Helper macro for calling `invariant` and constructing the format-string
   * arguments list.  Enabled in any build mode.
   */
#define SANDBOX_INVARIANT(cond, msg, ...) \
  sandbox::invariant(cond, FMT_STRING(msg), std::make_tuple(__VA_ARGS__))

  /**
   * Helper macro for calling `invariant` and constructing the format-string
   * arguments list.  Enabled only in debug builds.
   */
#define SANDBOX_DEBUG_INVARIANT(cond, msg, ...) \
  sandbox::invariant<DebugOnly>( \
    cond, FMT_STRING(msg), std::make_tuple(__VA_ARGS__))

  /**
   * Helper macro for calling `invariant` and constructing the format-string
   * arguments list.  Enabled only in release builds.
   */
#define SANDBOX_RELEASE_INVARIANT(cond, msg, ...) \
  sandbox::invariant<ReleaseOnly>( \
    cond, FMT_STRING(msg), std::make_tuple(__VA_ARGS__))

  /**
   * Helper that constructs a deleter from a C function, so that it can
   * be used with `std::unique_ptr`.
   */
  template<auto fn>
  using deleter_from_fn = std::integral_constant<decltype(fn), fn>;

  /**
   * Pointer from `malloc` that will be automatically `free`d.
   */
  template<typename T>
  using unique_c_ptr = std::unique_ptr<T, deleter_from_fn<::free>>;

  namespace internal
  {
    /**
     * Template that deduces the return type and argument types for a function
     * `signature<void(int, float)>::return_type` is `void` and
     * `signature<void(int, float)>::argument_type` is `std::tuple<int, float>`.
     */
    template<typename T>
    struct signature;

    /**
     * Specialisation for when the callee is a value.
     */
    template<typename R, typename... Args>
    struct signature<R(Args...)>
    {
      /**
       * The return type of the function whose type is being extracted.
       */
      using return_type = R;

      /**
       * A tuple type containing all of the argument types of the function
       * whose type is being extracted.
       */
      using argument_type = std::tuple<Args...>;
    };

    /**
     * Specification for when the callee is a reference.
     */
    template<typename R, typename... Args>
    struct signature<R (&)(Args...)>
    {
      /**
       * The return type of the function whose type is being extracted.
       */
      using return_type = R;

      /**
       * A tuple type containing all of the argument types of the function
       * whose type is being extracted.
       */
      using argument_type = std::tuple<Args...>;
    };
  }
}
