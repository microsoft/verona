// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../sched/action.h"
#include "../sched/cown.h"
#include "type_traits"

namespace verona::rt
{
  using namespace snmalloc;

  /***
   * This class wraps up a C++ implementation of a Verona closure.
   *
   * It lifts the methods from the inheriting class into the Verona descriptor
   * thus allowing the runtime to run the closure, and trace it for references
   * to regions, immutables or cowns.
   *
   * The C++ calling code is not expected to allocate this class, and the
   * instead should use Cown::schedule with this type as a parameter, so the
   * closure can be allocated as part of the runtime messages.
   **/
  template<class T>
  class VAction : public Action
  {
    friend class Cown;

  private:
    static void gc_trace(const Action* msg, ObjectStack& st)
    {
      (static_cast<const T*>(msg))->trace(st);
    }

    static void f(Action* msg)
    {
      auto t = static_cast<T*>(msg);
      t->f();

      // If action has a destructor tidy up the action.
      if constexpr (!std::is_trivially_destructible_v<T>)
      {
        t->~T();
      }
    }

    static const Action::Descriptor* desc()
    {
      static constexpr Action::Descriptor desc = {sizeof(T), f, gc_trace};

      return &desc;
    }

    void trace(ObjectStack&) const {}

  public:
    VAction() : Action(desc())
    {
      static_assert(
        std::is_base_of_v<Action, T>,
        "Template parameter must inherit from Action.");
    }

  private:
    /**
     * Placement new for allocating in already allocated memory
     *
     * The runtime will handle all allocation of the actions.
     **/
    void* operator new(size_t, VAction* obj)
    {
      return obj;
    }

    void operator delete(void*, VAction*) {}

    void* operator new(size_t) = delete;
    void* operator new[](size_t size) = delete;
    void operator delete[](void* p) = delete;
    void operator delete[](void* p, size_t sz) = delete;
  };
} // namespace verona::rt
