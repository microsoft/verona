// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include <cstdlib>
#include <variant>

/**
 * GCC and MSVC don't assume exhaustivity of enum classes.
 * Clang does.
 *
 * This macro is used to allow clang to continue to check exhaustivity assuming
 * it is a member of the enum class.
 **/
#if defined(__GNUC__) || defined(_MSC_VER)
#  define EXHAUSTIVE_SWITCH \
    default: \
      abort();
#else
#  define EXHAUSTIVE_SWITCH
#endif

template<typename O>
O truncate(size_t in)
{
  constexpr size_t number_of_bytes = sizeof(O);
  constexpr size_t bits_in_byte = 8;
  constexpr size_t one = 1;
  constexpr size_t mask = (one << (bits_in_byte * number_of_bytes)) - 1;
  if (in != (in & mask))
  {
    // truncation error
    std::abort();
  }

  return (in & mask);
}

/**
 * Combine multiple callable objects (e.g. lambdas) into a single one that
 * exposes all operator() implementations, as overloads.
 *
 * This is used to build an anonymous visitor that gets passed to std::visit.
 *
 * See C++ proposal P0051.
 */
template<typename... Ts>
struct overload : Ts...
{
  using Ts::operator()...;
};
template<typename... Ts>
overload(Ts...)->overload<Ts...>;

/**
 * Match a set of patterns to an std::variant.
 *
 * The pattern that matches the variant inner value best will be executed, where
 * "best" follows C++ overload resolution rules.
 *
 * decltype(auto) is used in place of just auto to allow the return type to be a
 * reference.
 */
template<typename... Vs, typename... Ts>
decltype(auto) match(const std::variant<Vs...>& value, Ts&&... patterns)
{
  return std::visit(overload{std::forward<Ts>(patterns)...}, value);
}

template<typename... Vs, typename... Ts>
decltype(auto) match(std::variant<Vs...>& value, Ts&&... patterns)
{
  return std::visit(overload{std::forward<Ts>(patterns)...}, value);
}
