// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../region/region.h"

#include <type_traits>

namespace verona::rt
{
  using namespace snmalloc;

  template<typename A>
  struct has_trace_possibly_iso
  {
  private:
    template<typename B>
    static auto test(ObjectStack* st)
      -> decltype(std::declval<B>().trace_possibly_iso(st), std::true_type());

    template<typename>
    static std::false_type test(...);

  public:
    static constexpr bool value =
      std::is_same_v<std::true_type, decltype(test<A>(nullptr))>;
  };

  template<typename A>
  struct has_notified
  {
  private:
    template<typename B>
    static auto test(Object* o)
      -> decltype(std::declval<B>().notified(o), std::true_type());

    template<typename>
    static std::false_type test(...);

  public:
    static constexpr bool value =
      std::is_same_v<std::true_type, decltype(test<A>(nullptr))>;
  };

  template<
    class T,
    RegionType region_type = RegionType::Trace,
    class Base = Object>
  class V : public Base
  {
  private:
    static_assert(
      std::is_same_v<Base, Object> ?
        region_type == RegionType::Trace || region_type == RegionType::Arena :
        region_type == RegionType::Cown);
    static_assert(
      std::is_same_v<Base, Object> || std::is_same_v<Base, Cown>,
      "V base must be Object or Cown");

    using RegionClass = typename RegionType_to_class<region_type>::T;

    static void gc_trace(const Object* o, ObjectStack* st)
    {
      ((T*)o)->trace(st);
    }

    static void gc_trace_possibly_iso(const Object* o, ObjectStack* st)
    {
      ((T*)o)->trace_possibly_iso(st);
    }

    static void gc_notified(Object* o)
    {
      ((T*)o)->notified(o);
    }

    static void gc_final(Object* o)
    {
      ((T*)o)->~T();
    }

    static const Descriptor* desc()
    {
      static constexpr Descriptor desc = {
        sizeof(T),
        gc_trace,
        has_trace_possibly_iso<T>::value ? gc_trace_possibly_iso : nullptr,
        std::is_trivially_destructible_v<T> ? nullptr : gc_final,
        has_notified<T>::value ? gc_notified : nullptr};

      return &desc;
    }

    void trace(ObjectStack*) {}

    // Dummy functions to make compiler happy.
    void trace_possibly_iso(ObjectStack*)
    {
      abort();
    }

    void notified(Object*)
    {
      abort();
    }

    static EpochMark get_alloc_epoch()
    {
      return Scheduler::alloc_epoch();
    }

  public:
    void* operator new(size_t)
    {
      if constexpr (std::is_same_v<Base, Object>)
        return RegionClass::template create<sizeof(T)>(
          ThreadAlloc::get(), desc());
      else
        return Cown::alloc<sizeof(T)>(
          ThreadAlloc::get(), desc(), get_alloc_epoch());
    }

    void* operator new(size_t, Alloc* alloc)
    {
      if constexpr (std::is_same_v<Base, Object>)
        return RegionClass::template create<sizeof(T)>(alloc, desc());
      else
        return Cown::alloc<sizeof(T)>(alloc, desc(), get_alloc_epoch());
    }

    void* operator new(size_t, Object* region)
    {
      if constexpr (std::is_same_v<Base, Object>)
        return RegionClass::template alloc<sizeof(T)>(
          ThreadAlloc::get(), region, desc());
      else
        return Cown::alloc<sizeof(T)>(
          ThreadAlloc::get(), desc(), get_alloc_epoch());
    }

    void* operator new(size_t, Alloc* alloc, Object* region)
    {
      if constexpr (std::is_same_v<Base, Object>)
        return RegionClass::template alloc<sizeof(T)>(alloc, region, desc());
      else
        return Cown::alloc<sizeof(T)>(alloc, desc(), get_alloc_epoch());
    }

    void operator delete(void*)
    {
      // Should not be called directly, present to allow calling if the
      // constructor throws an exception. The object lifetime is managed by the
      // region.
    }

    void operator delete(void*, size_t)
    {
      // Should not be called directly, present to allow calling if the
      // constructor throws an exception. The object lifetime is managed by the
      // region.
    }

    void operator delete(void*, Alloc*)
    {
      // Should not be called directly, present to allow calling if the
      // constructor throws an exception. The object lifetime is managed by the
      // region.
    }

    void operator delete(void*, Object*)
    {
      // Should not be called directly, present to allow calling if the
      // constructor throws an exception. The object lifetime is managed by the
      // region.
    }

    void operator delete(void*, Alloc*, Object*)
    {
      // Should not be called directly, present to allow calling if the
      // constructor throws an exception. The object lifetime is managed by the
      // region.
    }

    void* operator new[](size_t size) = delete;
    void operator delete[](void* p) = delete;
    void operator delete[](void* p, size_t sz) = delete;
  };

  // Cowns are not allocated inside regions, but we still need a RegionType.
  template<class T>
  using VCown = V<T, RegionType::Cown, Cown>;
} // namespace verona::rt
