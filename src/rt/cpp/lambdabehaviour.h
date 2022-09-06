// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "vbehaviour.h"
#include "vobject.h"

namespace verona::rt
{
  class EmptyCown : public VCown<EmptyCown>
  {
  public:
    EmptyCown() {}
  };

  template<class T>
  class LambdaBehaviour : public Behaviour
  {
    friend class Cown;
    friend class MultiMessage;

  private:
    T fn;

    static void f(Behaviour* msg)
    {
      auto t = static_cast<LambdaBehaviour<T>*>(msg);
      t->fn();

      if constexpr (!std::is_trivially_destructible_v<LambdaBehaviour<T>>)
      {
        t->~LambdaBehaviour<T>();
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
    LambdaBehaviour(T fn_) : Behaviour(desc()), fn(std::move(fn_)) {}
  };

  template<TransferOwnership transfer = NoTransfer, typename T>
  static void schedule_lambda(Cown* c, T f)
  {
    Cown::schedule<LambdaBehaviour<T>, transfer>(c, std::forward<T>(f));
  }

  template<TransferOwnership transfer = NoTransfer, typename T>
  static void schedule_lambda(size_t count, Cown** cowns, T f)
  {
    Cown::schedule<LambdaBehaviour<T>, transfer>(
      count, cowns, std::forward<T>(f));
  }

  template<TransferOwnership transfer = NoTransfer, typename T>
  static void schedule_lambda(size_t count, Request* requests, T f)
  {
    Cown::schedule<LambdaBehaviour<T>, transfer>(
      count, requests, std::forward<T>(f));
  }

  template<typename T>
  static void schedule_lambda(T f)
  {
    Cown* c = new EmptyCown();
    Cown::schedule<LambdaBehaviour<T>, YesTransfer>(c, std::forward<T>(f));
  }
} // namespace verona::rt
