// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../region/region.h"

#include <type_traits>

namespace verona::rt
{
  using namespace snmalloc;

  // These helpers are used to determine if various methods are provided by the
  // child class of V<>. They intentionally only check for the name of the
  // method, not for its precise signature.
  //
  // If for example a class C has a notified method with an incorrect
  // signature, `has_notified<C>` will still be true. However the
  // implementation of V<C> (in this case `gc_notified`) would not compile.
  //
  // This is better than ignoring methods with the right name but the wrong
  // signature.
  template<class T, class = void>
  struct has_notified_cpp : std::false_type
  {};
  template<class T>
  struct has_notified_cpp<T, std::void_t<decltype(&T::notified)>> : std::true_type
  {};

  template<class T, class = void>
  struct has_finaliser_cpp : std::false_type
  {};
  template<class T>
  struct has_finaliser_cpp<T, std::void_t<decltype(&T::finaliser)>> : std::true_type
  {};

  template<class T, class = void>
  struct has_trace_cpp : std::false_type
  {};
  template<class T>
  struct has_trace_cpp<T, std::void_t<decltype(&T::trace)>> : std::true_type
  {};

  template<class T>
  struct has_destructor_cpp
  {
    constexpr static bool value = !std::is_trivially_destructible_v<T>;
  };

  template<
    class T, class Base = Object>
  class VRep : public Base
  {
    /// Embedded C++ object
  public: 
    T contents;
  private:

    static void gc_trace(const Object* o, ObjectStack& st)
    {
      if constexpr (has_trace_cpp<T>::value)
        (((VRep*)o)->contents).trace(st);
    }

    static void gc_notified(Object* o)
    {
      if constexpr (has_notified_cpp<T>::value)
        (((VRep*)o)->contents).notified(o);
    }

    static void gc_final(Object* o, Object* region, ObjectStack& sub_regions)
    {
      if constexpr (has_finaliser_cpp<T>::value)
        (((VRep*)o)->contents).finaliser(region, sub_regions);
    }

    static void gc_destructor(Object* o)
    {
      (((VRep*)o)->contents).~T();
    }

  public:
    static const Descriptor* desc()
    {
      static constexpr Descriptor desc = {
        sizeof(VRep),
        gc_trace,
        has_finaliser_cpp<T>::value ? gc_final : nullptr,
        has_notified_cpp<T>::value ? gc_notified : nullptr,
        has_destructor_cpp<T>::value ? gc_destructor : nullptr};

      return &desc;
    }
  };

  template<
    class T,
    class Base = Object>
  class V
  {
    template<
      class T2,
      class Base2>
    friend bool operator!=(V<T2,Base2>, Object*);

    using VRep = VRep<T, Base>;
  //private:
  public:
    VRep* rep = nullptr;

    V(VRep* rep) : rep(rep) {}

  public:
    V() {}

    T* operator->() {
      return &(rep->contents);
    }

    operator Base*()
    {
      return rep;
    }

    operator Base*() const
    {
      return rep;
    }

    operator T*()
    {
      return &(rep->contents);
    }

    operator T*() const
    {
      return &(rep->contents);
    }

    const T* operator->() const {
      return &(rep->contents);
    }

    const T& operator*() const {
      return rep->contents;
    }

    T& operator*() {
      return rep->contents;
    }

    size_t id()
    {
      return rep->id();
    }
  };

  template<class T>
  using VCown = V<T, Cown>;

  template<class T>
  class VAlloc
  {
  public:
    template<typename... Args>
    static VCown<T> make_cown(Alloc* alloc, Args... args)
    {
      using VRep = VRep<T, Cown>;
      VRep* object = (VRep*)Cown::alloc(alloc, VRep::desc(), Scheduler::alloc_epoch());
      
      new (&(object->contents)) T(std::forward<Args>(args)...);

      return VCown<T>(object);
    }

    template<typename... Args>
    static VCown<T> make_cown(Args... args)
    {
      return make_cown(ThreadAlloc::get(), std::forward<Args>(args)...);
    }

    template<class RegionClass, typename... Args>
    static V<T> make(Alloc* alloc, Args... args)
    {
      VRep<T>*  object = (VRep<T>*)RegionClass::template create<sizeof(VRep<T>)>(alloc, VRep<T>::desc());

      // Initialise C++ object contents
      new (&(object->contents)) T(std::forward<Args>(args)...);

      return V<T>(object);
    }

    template<typename... Args>
    static V<T> make_arena(Alloc* alloc, Args... args)
    {
      return make<RegionArena, Args...>(alloc, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static V<T> make_trace(Alloc* alloc, Args... args)
    {
      return make<RegionTrace, Args...>(alloc, std::forward<Args>(args)...);
    }

    template<class RegionClass, typename... Args>
    static V<T> make_in(Object* region, Alloc* alloc, Args... args)
    {
      // Build object in correct region
      VRep<T>* object = (VRep<T>*)RegionClass::template alloc<sizeof(VRep<T>)>(alloc, region, VRep<T>::desc());
      // Initialise C++ object contents
      new (&(object->contents)) T(std::forward<Args>(args)...);
      return V<T>(object);
    }

    template<typename... Args>
    static V<T> make_in_arena(Object* region, Alloc* alloc, Args... args)
    {
      return make_in<RegionArena, Args...>(region, alloc, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static V<T> make_in_trace(Object* region, Alloc* alloc, Args... args)
    {
      return make_in<RegionTrace, Args...>(region, alloc, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static V<T> make_in_arena(Object* region, Args... args)
    {
      return make_in_arena<Args...>(region, ThreadAlloc::get(), std::forward<Args>(args)...);
    }

    template<typename... Args>
    static V<T> make_in_trace(Object* region, Args... args)
    {
      return make_in_trace<Args...>(region, ThreadAlloc::get(), std::forward<Args>(args)...);
    }

    template<typename... Args>
    static V<T> make_arena(Args... args)
    {
      return make_arena<Args...>(ThreadAlloc::get(), std::forward<Args>(args)...);
    }

    template<typename... Args>
    static V<T> make_trace(Args... args)
    {
      return make_trace<Args...>(ThreadAlloc::get(), std::forward<Args>(args)...);
    }
  };

  template<
    class T,
    class Base>
  static bool operator!=(V<T, Base> v, Object* o)
  {
    return v.rep != o;
  }
} // namespace verona::rt
