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
  struct has_notified : std::false_type
  {};
  template<class T>
  struct has_notified<T, std::void_t<decltype(&T::notified)>> : std::true_type
  {};

  template<class T, class = void>
  struct has_finaliser : std::false_type
  {};
  template<class T>
  struct has_finaliser<T, std::void_t<decltype(&T::finaliser)>> : std::true_type
  {};

  template<class T>
  struct has_destructor
  {
    constexpr static bool value = !std::is_trivially_destructible_v<T>;
  };

  /**
   * Common base class for V and VCown to build descriptors
   * from C++ objects using compile time reflection.
   */
  template<class T, class Base = Object>
  class VBase : public Base
  {
  private:
    static void gc_trace(const Object* o, ObjectStack& st)
    {
      ((T*)o)->trace(st);
    }

    static void gc_notified(Object* o)
    {
      if constexpr (has_notified<T>::value)
        ((T*)o)->notified(o);
      else
      {
        UNUSED(o);
      }
    }

    static void gc_final(Object* o, Object* region, ObjectStack& sub_regions)
    {
      if constexpr (has_finaliser<T>::value)
        ((T*)o)->finaliser(region, sub_regions);
      else
      {
        UNUSED(o);
        UNUSED(region);
        UNUSED(sub_regions);
      }
    }

    static void gc_destructor(Object* o)
    {
      ((T*)o)->~T();
    }

    void trace(ObjectStack&) {}

  public:
    VBase() : Base() {}

    static Descriptor* desc()
    {
      static Descriptor desc = {vsizeof<T>,
                                gc_trace,
                                has_finaliser<T>::value ? gc_final : nullptr,
                                has_notified<T>::value ? gc_notified : nullptr,
                                has_destructor<T>::value ? gc_destructor :
                                                           nullptr};

      return &desc;
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

    void operator delete(void*, Alloc&)
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

    void operator delete(void*, Alloc&, Object*)
    {
      // Should not be called directly, present to allow calling if the
      // constructor throws an exception. The object lifetime is managed by the
      // region.
    }

    void* operator new[](size_t size) = delete;
    void operator delete[](void* p) = delete;
    void operator delete[](void* p, size_t sz) = delete;
  };

  /**
   * Converts a C++ class into a Verona Object
   *
   * Will fill the Verona descriptor with relevant fields.
   */
  template<class T, RegionType region_type = RegionType::Trace>
  class V : public VBase<T, Object>
  {
    using RegionClass = typename RegionType_to_class<region_type>::T;

  public:
    V() : VBase<T, Object>() {}

    void* operator new(size_t)
    {
      return RegionClass::template create<vsizeof<T>>(
        ThreadAlloc::get(), VBase<T, Object>::desc());
    }

    void* operator new(size_t, Alloc& alloc)
    {
      return RegionClass::template create<vsizeof<T>>(
        alloc, VBase<T, Object>::desc());
    }

    void* operator new(size_t, Object* region)
    {
      return RegionClass::template alloc<vsizeof<T>>(
        ThreadAlloc::get(), region, VBase<T, Object>::desc());
    }

    void* operator new(size_t, Alloc& alloc, Object* region)
    {
      return RegionClass::template alloc<vsizeof<T>>(
        alloc, region, VBase<T, Object>::desc());
    }
  };

  /**
   * Converts a C++ class into a Verona Cown
   *
   * Will fill the Verona descriptor with relevant fields.
   */
  template<class T>
  class VCown : public VBase<T, Cown>
  {
  public:
    VCown() : VBase<T, Cown>() {}

    void* operator new(size_t)
    {
      return Object::register_object(
        ThreadAlloc::get().alloc<vsizeof<T>>(), VBase<T, Cown>::desc());
    }

    void* operator new(size_t, Alloc& alloc)
    {
      return Object::register_object(
        alloc.alloc<vsizeof<T>>(), VBase<T, Cown>::desc());
    }
  };
} // namespace verona::rt
