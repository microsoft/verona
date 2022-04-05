// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <functional>
#include <tuple>
#include <utility>
#include <verona.h>

/**
 * Class for staging the when creation.
 *
 * Do not call directly use `when`
 * 
 * This provides an operator << to apply the closure.  This allows the 
 * argument order to be more sensible, as variadic arguments have to be last.
 *
 *   when (cown1, ..., cownn) << closure;
 * 
 * Allows the variadic number of cowns to occur before the closure.
 */
template<typename... Args>
class When
{
  template<typename... Args2>
  friend When<Args2...> when(Args2&&... args);

  std::tuple<Args...> cown_tuple;

  /**
   * This uses template programming to turn the std::tuple into a C style
   * stack allocated array.
   * The index template parameter is used to perform each the assignment for
   * each index.
   */
  template<size_t index>
  void array_assign(Cown** array)
  {
    if constexpr (index >= sizeof...(Args))
    {
      return;
    }
    else
    {
      array[index] = std::get<index>(cown_tuple);
      array_assign<index + 1>(array);
    }
  }

  template<typename... Ts>
  When(Ts&&... args) : cown_tuple(std::forward<Ts>(args)...)
  {}

public:
  template<typename F>
  void operator<<(F&& f)
  {
    verona::rt::Cown* cowns[sizeof...(Args)];
    array_assign<0>(cowns);

    verona::rt::schedule_lambda(
      sizeof...(Args),
      cowns,
      [f = std::forward<F>(f), cown_tuple = cown_tuple]() mutable {
        std::apply(f, cown_tuple);
      });
  }
};

/**
 * Implements a Verona-like `when` statement.
 *
 * Uses `<<` to apply the closure.
 */
template<typename... Args>
When<Args...> when(Args&&... args)
{
  return When<Args...>(std::forward<Args>(args)...);
}