// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "vbehaviour.h"
#include "vobject.h"

namespace verona::rt
{
  // using namespace snmalloc;

  class EmptyCown : public VCown<EmptyCown>
  {
  public:
    EmptyCown() {}
  };

  template<class T>
  class LambdaBehaviour : public Behaviour
  {
    friend class Cown;

  private:
    T fn;

    static void f(Behaviour* msg)
    {
      auto t = static_cast<LambdaBehaviour<T>*>(msg);
      t->fn();

      if constexpr (!std::is_trivially_destructible_v<T>)
      {
        t->~T();
      }
    }

    static const Behaviour::Descriptor* desc()
    {
      static constexpr Behaviour::Descriptor desc = {
        sizeof(LambdaBehaviour<T>), f, NULL};

      return &desc;
    }

    void* operator new(size_t, LambdaBehaviour* obj)
    {
      return obj;
    }

    void operator delete(void*, LambdaBehaviour*) {}

    void* operator new(size_t) = delete;
    void* operator new[](size_t size) = delete;
    void operator delete[](void* p) = delete;
    void operator delete[](void* p, size_t sz) = delete;

  public:
    LambdaBehaviour(T fn_) : Behaviour(desc()), fn(fn_) {}
  };

  template<typename T>
  static void scheduleLambda(Cown* c, T f)
  {
    Cown::schedule<LambdaBehaviour<T>>(c, f);
  }

  template<typename T>
  static void scheduleLambda(T f)
  {
    Cown* c = new EmptyCown();
    Cown::schedule<LambdaBehaviour<T>>(c, f);
    Cown::release(ThreadAlloc::get(), c);
  }
} // namespace verona::rt
