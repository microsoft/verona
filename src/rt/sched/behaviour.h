// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"

#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;
  class Cown;

  struct Request
  {
    Cown* _cown;

    static const uintptr_t READ_FLAG = 1;

    Request() : _cown(nullptr) {}
    Request(Cown* cown) : _cown(cown) {}

    Cown* cown()
    {
      return (Cown*)((uintptr_t)_cown & ~READ_FLAG);
    }

    bool is_read()
    {
      return ((uintptr_t)_cown & READ_FLAG);
    }

    static Request write(Cown* cown)
    {
      return Request(cown);
    }

    static Request read(Cown* cown)
    {
      return Request((Cown*)((uintptr_t)cown | READ_FLAG));
    }
  };

  /**
   * This class represents the closure run when all the cowns required have
   * been acquired.
   *
   * It provides two methods in its descriptor:
   *  - A run method, `f`.
   *  - A trace method.
   *
   * The trace is used during leak detection to allow the closure state to be
   * scanned.
   **/
  class Behaviour
  {
    friend class Cown;
    friend class MultiMessage;

  public:
    struct alignas(descriptor_alignment) Descriptor
    {
      using Function = void (*)(Behaviour*);
      using TraceFunction = void (*)(const Behaviour*, ObjectStack&);

      size_t size;

      /**
       * Run the body of a "when". If the behaviour contains any non-trivial
       * state, then the last thing f should do is finalise that state. The
       * behaviour itself will deallocated by the runtime.
       **/
      Function f;

      /**
       * Trace the reachable objects from this behaviour.
       **/
      TraceFunction trace;

      static void empty_behaviour_f(Behaviour*) {}
      static void empty_behaviour_trace(const Behaviour*, ObjectStack&) {}
      static const Descriptor* empty()
      {
        static const Descriptor desc = {
          sizeof(Behaviour), empty_behaviour_f, empty_behaviour_trace};
        return &desc;
      }
    };

  protected:
    const Descriptor* descriptor;

  public:
    Behaviour(const Descriptor* desc) : descriptor(desc) {}

  protected:
    inline size_t size() const
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

    inline const Descriptor* get_descriptor() const
    {
      return descriptor;
    }
  };
} // namespace verona::rt
