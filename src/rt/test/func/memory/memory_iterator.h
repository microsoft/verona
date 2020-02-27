// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "memory.h"

#include <unordered_set>

namespace memory_iterator
{
  /**
   * Helper for iterator tests that allocates objects into the region whose
   * iso object is `o`, and then maintains unordered sets to ensure the
   * the iterators traverse over all objects, all trivial objects, and all
   * non-trivial objects.
   **/
  template<class T, class... Rest>
  void test_iterator_insert(
    Alloc* alloc,
    Object* o,
    std::unordered_set<Object*>& all,
    std::unordered_set<Object*>& trivial,
    std::unordered_set<Object*>& non_trivial)
  {
    UNUSED(trivial);
    UNUSED(non_trivial);

    Object* t = new (alloc, o) T;
    all.insert(t);

    if constexpr (
      has_trace_possibly_iso<T>::value || !std::is_trivially_destructible_v<T>)
      non_trivial.insert(t);
    else
      trivial.insert(t);

    if constexpr (sizeof...(Rest) > 0)
      test_iterator_insert<Rest...>(alloc, o, all, trivial, non_trivial);
  }

  template<RegionType region_type>
  void test_simple()
  {
    using RegionClass = typename RegionType_to_class<region_type>::T;
    using C = C1<region_type>;
    using F = F1<region_type>;

    // Only one non-trivial object.
    {
      auto* o = new F;
      auto* reg = RegionClass::get(o);

      for (auto p : *reg)
      {
        UNUSED(p);
        assert(p == o);
      }

      for (auto n_it = reg->template begin<RegionBase::Trivial>();
           n_it != reg->template end<RegionBase::Trivial>();
           ++n_it)
      {
        assert(0); // unreachable
      }

      for (auto f_it = reg->template begin<RegionBase::NonTrivial>();
           f_it != reg->template end<RegionBase::NonTrivial>();
           ++f_it)
      {
        assert(*f_it == o);
      }

      Region::release(ThreadAlloc::get(), o);
    }

    // Only one trivial object.
    {
      auto* o = new C;
      auto* reg = RegionClass::get(o);

      for (auto p : *reg)
      {
        UNUSED(p);
        assert(p == o);
      }

      for (auto n_it = reg->template begin<RegionBase::Trivial>();
           n_it != reg->template end<RegionBase::Trivial>();
           ++n_it)
      {
        assert(*n_it == o);
      }

      for (auto f_it = reg->template begin<RegionBase::NonTrivial>();
           f_it != reg->template end<RegionBase::NonTrivial>();
           ++f_it)
      {
        assert(0); // unreachable
      }

      Region::release(ThreadAlloc::get(), o);
    }
  }

  /**
   * Tests the region iterator.
   **/
  template<RegionType region_type>
  void test_iterator()
  {
    if constexpr (region_type == RegionType::Trace)
    {
      using C = C3<region_type>;
      using F = F3<region_type>;

      C* oc = nullptr;
      F* of = nullptr;
      std::unordered_set<Object*> s1; // all objects
      std::unordered_set<Object*> s2; // trivial
      std::unordered_set<Object*> s3; // non-trivial
      std::unordered_set<Object*> s4; // to be garbage collected
      auto* alloc = ThreadAlloc::get();

      auto* r = new (alloc) C;
      auto* reg = RegionTrace::get(r);
      s1.insert(r);
      s2.insert(r); // Note: might need to change this if o's type is changed.

      // Add some objects. Some of them will be part of the object graph, while
      // others will be GC'd.
      oc = new (alloc, r) C;
      s1.insert(oc);
      s2.insert(oc);
      s4.insert(oc);

      oc = new (alloc, r) C;
      s1.insert(oc);
      s2.insert(oc);
      r->c1 = oc;

      of = new (alloc, r) F;
      s1.insert(of);
      s3.insert(of);
      r->f1 = of;

      oc = new (alloc, r) C;
      s1.insert(oc);
      s2.insert(oc);
      r->f1->c1 = oc;

      of = new (alloc, r) F;
      s1.insert(of);
      s3.insert(of);
      r->c1->f1 = of;

      oc = new (alloc, r) C;
      s1.insert(oc);
      s2.insert(oc);
      s4.insert(oc);

      of = new (alloc, r) F;
      s1.insert(of);
      s3.insert(of);
      s4.insert(of);

      of = new (alloc, r) F;
      s1.insert(of);
      s3.insert(of);
      s4.insert(of);

      // Sanity check.
      assert(s1.size() == s2.size() + s3.size());

      // Now check that we iterated over everything.
      for (auto p : *reg)
      {
        assert(s1.count(p));
        s1.erase(p);
      }
      assert(s1.empty());

      for (auto n_it = reg->template begin<RegionBase::Trivial>();
           n_it != reg->template end<RegionBase::Trivial>();
           ++n_it)
      {
        assert(s2.count(*n_it));
        s2.erase(*n_it);
      }
      assert(s2.empty());

      for (auto f_it = reg->template begin<RegionBase::NonTrivial>();
           f_it != reg->template end<RegionBase::NonTrivial>();
           ++f_it)
      {
        assert(s3.count(*f_it));
        s3.erase(*f_it);
      }
      assert(s3.empty());

      // Run a GC.
      assert(Region::debug_size(r) == 9);
      RegionTrace::gc(alloc, r);
      assert(Region::debug_size(r) == 5);

      // Check that we didn't collect anything we shouldn't have.
      for (auto p : *reg)
      {
        UNUSED(p);
        assert(!s4.count(p));
      }

      Region::release(alloc, r);
      snmalloc::current_alloc_pool()->debug_check_empty();
      assert(live_count == 0);
    }
    else if constexpr (region_type == RegionType::Arena)
    {
      using C = C1<region_type>;
      using F = F1<region_type>;
      using LC = LargeC2<region_type>;
      using LF = LargeF2<region_type>;
      using XC = XLargeC2<region_type>;
      using XF = XLargeF2<region_type>;

      std::unordered_set<Object*> s1; // all objects
      std::unordered_set<Object*> s2; // trivial
      std::unordered_set<Object*> s3; // non-trivial
      auto* alloc = ThreadAlloc::get();

      auto* o = new (alloc) XF;
      auto* reg = RegionArena::get(o);
      s1.insert(o);
      s3.insert(o); // Note: might need to change this if o's type is changed.

      // Put a few objects into the large object ring.
      test_iterator_insert<XC, XC, XC, XF, XC, XF, XF, XC>(
        alloc, o, s1, s2, s3);

      // Now a bunch of objects into an arena.
      test_iterator_insert<C, C, C, F, F, F>(alloc, o, s1, s2, s3);

      // Force a new arena, then only add trivial objects.
      test_iterator_insert<LC, C, C, C, C>(alloc, o, s1, s2, s3);

      // // Force a new arena, then only non-trivial objects.
      test_iterator_insert<LF, F, F, F, F>(alloc, o, s1, s2, s3);

      // // And now a chain of arenas.
      test_iterator_insert<LF, LF, LC, LC>(alloc, o, s1, s2, s3);

      // Sanity check.
      assert(s1.size() == s2.size() + s3.size());

      // Now check that we iterated over everything.
      for (auto p : *reg)
      {
        assert(s1.count(p));
        s1.erase(p);
      }
      assert(s1.empty());

      for (auto n_it = reg->template begin<RegionBase::Trivial>();
           n_it != reg->template end<RegionBase::Trivial>();
           ++n_it)
      {
        assert(s2.count(*n_it));
        s2.erase(*n_it);
      }
      assert(s2.empty());

      for (auto f_it = reg->template begin<RegionBase::NonTrivial>();
           f_it != reg->template end<RegionBase::NonTrivial>();
           ++f_it)
      {
        assert(s3.count(*f_it));
        s3.erase(*f_it);
      }
      assert(s3.empty());

      Region::release(alloc, o);
      snmalloc::current_alloc_pool()->debug_check_empty();
      assert(live_count == 0);
    }
  }

  void run_test()
  {
    test_simple<RegionType::Trace>();
    test_simple<RegionType::Arena>();

    test_iterator<RegionType::Trace>();
    test_iterator<RegionType::Arena>();
  }
}