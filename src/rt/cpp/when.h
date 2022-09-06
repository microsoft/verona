// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "cown.h"

#include <functional>
#include <tuple>
#include <utility>
#include <verona.h>

namespace verona::cpp
{
  using namespace verona::rt;

  /**
   * Used to track the type of access request by embedding const into
   * the type T, or not having const.
   */
  template<typename T>
  class Access
  {
    ActualCown<std::remove_const_t<T>>* t;

  public:
    Access(const cown_ptr<T>& c) : t(c.allocated_cown) {}

    template<typename... Args>
    friend class When;
  };

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
    template<class T>
    struct is_read_only : std::false_type
    {};
    template<class T>
    struct is_read_only<Access<const T>> : std::true_type
    {};

    // Note only requires friend when Args2 == Args
    // but C++ doesn't like this.
    template<typename... Args2>
    friend auto when(Args2&&... args);

    /**
     * Internally uses AcquiredCown.  The cown is only acquired after the
     * behaviour is scheduled.
     */
    std::tuple<Access<Args>...> cown_tuple;

    /**
     * This uses template programming to turn the std::tuple into a C style
     * stack allocated array.
     * The index template parameter is used to perform each the assignment for
     * each index.
     */
    template<size_t index = 0>
    void array_assign(Request* requests)
    {
      if constexpr (index >= sizeof...(Args))
      {
        return;
      }
      else
      {
        auto p = std::get<index>(cown_tuple);
        if constexpr (is_read_only<decltype(p)>())
          requests[index] = Request::read(p.t);
        else
          requests[index] = Request::write(p.t);
        assert(requests[index].cown() != nullptr);
        array_assign<index + 1>(requests);
      }
    }

    When(Access<Args>... args) : cown_tuple(args...) {}

    /**
     * Converts a single `cown_ptr` into a `acquired_cown`.
     *
     * Needs to be a separate function for the template parameter to work.
     */
    template<typename C>
    static acquired_cown<C> access_to_acquired(Access<C> c)
    {
      assert(c.t != nullptr);
      return acquired_cown<C>(*c.t);
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
        verona::rt::Request requests[sizeof...(Args)];
        array_assign(requests);

        verona::rt::schedule_lambda(
          sizeof...(Args),
          requests,
          [f = std::forward<F>(f), cown_tuple = cown_tuple]() mutable {
            /// Effectively converts ActualCown<T>... to
            /// acquired_cown... .
            auto lift_f = [f =
                             std::forward<F>(f)](Access<Args>... args) mutable {
              f(access_to_acquired<Args>(args)...);
            };

            std::apply(lift_f, cown_tuple);
          });
      }
    }
  };

  /**
   * Template deduction guide for when.
   */
  template<typename... Args>
  When(Access<Args>...)->When<Args...>;

  /**
   * Template deduction guide for Access.
   */
  template<typename T>
  Access(const cown_ptr<T>&)->Access<T>;

  /**
   * Implements a Verona-like `when` statement.
   *
   * Uses `<<` to apply the closure.
   *
   * This should really take a type of
   *   ((cown_ptr<A1>& | cown_ptr<A1>&&)...
   * To get the universal reference type to work, we can't
   * place this constraint on it directly, as it needs to be
   * on a type argument.
   */
  template<typename... Args>
  auto when(Args&&... args)
  {
    return When(Access(args)...);
  }
} // namespace verona::cpp