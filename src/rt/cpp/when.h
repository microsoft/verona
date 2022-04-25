// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "cown.h"

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
  // Note only requires friend when Args2 == Args
  // but C++ doesn't like this.
  template<typename... Args2>
  friend When<Args2...> when(Args2... args);

  /**
   * Internally uses AcquiredCown.  The cown is only acquired after the
   * behaviour is scheduled.
   */
  std::tuple<Args...> cown_tuple;

  /**
   * This uses template programming to turn the std::tuple into a C style
   * stack allocated array.
   * The index template parameter is used to perform each the assignment for
   * each index.
   */
  template<size_t index = 0>
  void array_assign(Cown** array)
  {
    if constexpr (index >= sizeof...(Args))
    {
      return;
    }
    else
    {
      auto p = std::get<index>(cown_tuple);
      array[index] = p.underlying_cown();
      assert(array[index] != nullptr);
      array_assign<index + 1>(array);
    }
  }

  template<typename... Ts>
  When(Ts... args) : cown_tuple(args...)
  {
    static_assert(
      std::conjunction_v<std::is_base_of<cown_ptr_base, Ts>...>,
      "Not a cown_ptr");
  }

  /**
   * Converts a single `cown_ptr` into a `acquired_cown`.
   *
   * Needs to be a separate function for the template parameter to work.
   */
  template<typename C>
  static acquired_cown<C> cown_ptr_to_acquired(cown_ptr<C> c)
  {
    return acquired_cown<C>(c);
  }

public:
  /**
   * Applies the closure to schedule the behaviour on the set of cowns.
   */
  template<typename F>
  void operator<<(F&& f)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      verona::rt::schedule_lambda(std::forward<F>(f));
    }
    else
    {
      verona::rt::Cown* cowns[sizeof...(Args)];
      array_assign(cowns);

      verona::rt::schedule_lambda(
        sizeof...(Args),
        cowns,
        [f = std::forward<F>(f), cown_tuple = cown_tuple]() mutable {
          /// Effectively converts cown_ptr... to acquired_cown... .
          auto lift_f = [f = std::forward<F>(f)](Args... args) mutable {
            f(cown_ptr_to_acquired(args)...);
          };

          std::apply(lift_f, cown_tuple);
        });
    }
  }
};

/**
 * Implements a Verona-like `when` statement.
 *
 * Uses `<<` to apply the closure.
 */
template<typename... Args>
When<Args...> when(Args... args)
{
  return When<Args...>(args...);
}