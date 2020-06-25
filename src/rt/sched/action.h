// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"

#include <snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;
  class Cown;

  /**
   * This class represents the action run when all the cowns required have been
   * acquired.
   *
   * It provides two methods in its descriptor
   *  - a run method, f.
   *  - a trace method.
   *
   * The trace is used during leak detection to allow the actions state
   * (closure) to be scanned.
   **/
  class Action
  {
    friend class Cown;
    friend class MultiMessage;

  public:
    struct alignas(descriptor_alignment) Descriptor
    {
      using Function = void (*)(Action*);
      using TraceFunction = void (*)(const Action*, ObjectStack&);

      size_t size;

      /**
       * The single function that is run during a when clause.
       *
       * If the action contains any non-trivial state, then the last thing
       * f should do is finalise that state.  The action itself will deallocated
       * by the runtime.
       **/
      Function f;

      /**
       * Trace the reachable objects from this action.
       **/
      TraceFunction trace;
    };

  protected:
    const Descriptor* descriptor;

  public:
    Action(const Descriptor* desc) : descriptor(desc) {}

  protected:
    inline size_t size()
    {
      return get_descriptor()->size;
    }

    inline void f()
    {
      return get_descriptor()->f(this);
    }

    inline void trace(ObjectStack& st)
    {
      get_descriptor()->trace(this, st);
    }

    inline const Descriptor* get_descriptor()
    {
      return descriptor;
    }
  };
} // namespace verona::rt
