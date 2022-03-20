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

  template<typename T>
  struct LambdaBehaviourPackedArgs;

  template<class T>
  class LambdaBehaviour : public Behaviour
  {
    friend class Cown;
    friend struct LambdaBehaviourPackedArgs<T>;

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

  template<typename T>
  struct LambdaBehaviourPackedArgs
  {
    using type = LambdaBehaviour<T>;

    LambdaBehaviour<T>* be;
    Cown** cowns;
    size_t count;

    LambdaBehaviourPackedArgs(T&& fn, Cown** cowns_, size_t count_)
    : cowns(cowns_), count(count_)
    {
      auto& alloc = ThreadAlloc::get();
      be = new ((LambdaBehaviour<T>*)alloc.alloc<sizeof(LambdaBehaviour<T>)>())
        LambdaBehaviour<T>(std::move(fn));
    }
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

  template<typename T>
  static void schedule_lambda(T f)
  {
    Cown* c = new EmptyCown();
    Cown::schedule<LambdaBehaviour<T>, YesTransfer>(c, std::forward<T>(f));
  }

  template<typename T>
  static void is_packed_args(LambdaBehaviourPackedArgs<T> args)
  {
    (void)args;
  }

#ifdef ACQUIRE_ALL
  template<TransferOwnership transfer = NoTransfer, typename... Args>
  static void schedule_lambda_many(Args&&... args)
  {
    // Hack to enforce the right variadic template types are PackedArg
    ([&](auto&& input) { is_packed_args(input); }(std::forward<Args>(args)),
     ...);

    Cown::schedule_many<transfer>(std::forward<Args>(args)...);
  }
#endif
} // namespace verona::rt
