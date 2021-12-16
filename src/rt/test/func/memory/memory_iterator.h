// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

#include <unordered_set>

namespace memory_iterator
{
  static constexpr uintptr_t FINALISER_MASK = 1 << 1;
  /**
   * Helper for iterator tests that allocates objects into the region whose
   * iso object is `o`, and then maintains unordered sets to ensure the
   * the iterators traverse over all objects, all trivial objects, and all
   * non-trivial objects.
   **/
  template<class T, class... Rest>
  void test_iterator_insert(
    std::unordered_set<Object*>& all,
    std::unordered_set<Object*>& trivial,
    std::unordered_set<Object*>& non_trivial)
  {
    UNUSED(trivial);
    UNUSED(non_trivial);

    Object* t = new T;
    all.insert(t);

    if constexpr (!std::is_trivially_destructible_v<T>)
      non_trivial.insert(t);
    else
      trivial.insert(t);

    if constexpr (sizeof...(Rest) > 0)
      test_iterator_insert<Rest...>(all, trivial, non_trivial);
  }

  template<RegionType region_type>
  void test_simple()
  {
    using RegionClass = typename RegionType_to_class<region_type>::T;
    using C = C1;
    using F = F1;

    {
      auto o = new (region_type) F;
      // Only one non-trivial object.
      {
        UsingRegion r(o);
        auto* reg = RegionClass::get(o);

        for (auto p : *reg)
        {
          UNUSED(p);
          if constexpr (region_type == RegionType::Rc)
          {
            p = (Object*)(((uintptr_t)p) & ~FINALISER_MASK);
          }
          check(p == o);
        }

        for (auto n_it = reg->template begin<RegionBase::Trivial>();
             n_it != reg->template end<RegionBase::Trivial>();
             ++n_it)
        {
          check(0); // unreachable
        }

        for (auto f_it = reg->template begin<RegionBase::NonTrivial>();
             f_it != reg->template end<RegionBase::NonTrivial>();
             ++f_it)
        {
          if constexpr (region_type == RegionType::Rc)
          {
            auto p = (Object*)(((uintptr_t)*f_it) & ~FINALISER_MASK);
            check(p == o);
          }
          else
          {
            check(*f_it == o);
          }
        }
      }
      region_release(o);
    }
    // Only one trivial object.
    {
      auto o = new (region_type) C;
      {
        UsingRegion r(o);
        auto* reg = RegionClass::get(o);

        for (auto p : *reg)
        {
          UNUSED(p);
          check(p == o);
        }

        for (auto n_it = reg->template begin<RegionBase::Trivial>();
             n_it != reg->template end<RegionBase::Trivial>();
             ++n_it)
        {
          check(*n_it == o);
        }

        for (auto f_it = reg->template begin<RegionBase::NonTrivial>();
             f_it != reg->template end<RegionBase::NonTrivial>();
             ++f_it)
        {
          check(0); // unreachable
        }
      }
      region_release(o);
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
      using C = C3;
      using F = F3;

      C* oc = nullptr;
      F* of = nullptr;
      std::unordered_set<Object*> s1; // all objects
      std::unordered_set<Object*> s2; // trivial
      std::unordered_set<Object*> s3; // non-trivial
      std::unordered_set<Object*> s4; // to be garbage collected

      auto r = new (region_type) C;
      {
        UsingRegion r2(r);
        auto* reg = RegionTrace::get(r);
        s1.insert(r);
        s2.insert(r); // Note: might need to change this if o's type is changed.

        // Add some objects. Some of them will be part of the object graph,
        // while others will be GC'd.
        oc = new C;
        s1.insert(oc);
        s2.insert(oc);
        s4.insert(oc);

        oc = new C;
        s1.insert(oc);
        s2.insert(oc);
        r->c1 = oc;

        of = new F;
        s1.insert(of);
        s3.insert(of);
        r->f1 = of;

        oc = new C;
        s1.insert(oc);
        s2.insert(oc);
        r->f1->c1 = oc;

        of = new F;
        s1.insert(of);
        s3.insert(of);
        r->c1->f1 = of;

        oc = new C;
        s1.insert(oc);
        s2.insert(oc);
        s4.insert(oc);

        of = new F;
        s1.insert(of);
        s3.insert(of);
        s4.insert(of);

        of = new F;
        s1.insert(of);
        s3.insert(of);
        s4.insert(of);

        // Sanity check.
        check(s1.size() == s2.size() + s3.size());

        // Now check that we iterated over everything.
        for (auto p : *reg)
        {
          check(s1.count(p));
          s1.erase(p);
        }
        check(s1.empty());

        for (auto n_it = reg->template begin<RegionBase::Trivial>();
             n_it != reg->template end<RegionBase::Trivial>();
             ++n_it)
        {
          check(s2.count(*n_it));
          s2.erase(*n_it);
        }
        check(s2.empty());

        for (auto f_it = reg->template begin<RegionBase::NonTrivial>();
             f_it != reg->template end<RegionBase::NonTrivial>();
             ++f_it)
        {
          check(s3.count(*f_it));
          s3.erase(*f_it);
        }
        check(s3.empty());

        // Run a GC.
        check(debug_size() == 9);
        region_collect();
        check(debug_size() == 5);

        // Check that we didn't collect anything we shouldn't have.
        for (auto p : *reg)
        {
          UNUSED(p);
          check(!s4.count(p));
        }
      }

      region_release(r);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
      check(live_count == 0);
    }
    else if constexpr (region_type == RegionType::Arena)
    {
      using C = C1;
      using F = F1;
      using LC = LargeC2;
      using LF = LargeF2;
      using XC = XLargeC2;
      using XF = XLargeF2;

      std::unordered_set<Object*> s1; // all objects
      std::unordered_set<Object*> s2; // trivial
      std::unordered_set<Object*> s3; // non-trivial

      auto o = new (region_type) XF;
      {
        UsingRegion r2(o);
        auto* reg = RegionArena::get(o);
        s1.insert(o);
        s3.insert(o); // Note: might need to change this if o's type is changed.

        // Put a few objects into the large object ring.
        test_iterator_insert<XC, XC, XC, XF, XC, XF, XF, XC>(s1, s2, s3);

        // Now a bunch of objects into an arena.
        test_iterator_insert<C, C, C, F, F, F>(s1, s2, s3);

        // Force a new arena, then only add trivial objects.
        test_iterator_insert<LC, C, C, C, C>(s1, s2, s3);

        // // Force a new arena, then only non-trivial objects.
        test_iterator_insert<LF, F, F, F, F>(s1, s2, s3);

        // // And now a chain of arenas.
        test_iterator_insert<LF, LF, LC, LC>(s1, s2, s3);

        // Sanity check.
        check(s1.size() == s2.size() + s3.size());

        // Now check that we iterated over everything.
        for (auto p : *reg)
        {
          check(s1.count(p));
          s1.erase(p);
        }
        check(s1.empty());

        for (auto n_it = reg->template begin<RegionBase::Trivial>();
             n_it != reg->template end<RegionBase::Trivial>();
             ++n_it)
        {
          check(s2.count(*n_it));
          s2.erase(*n_it);
        }
        check(s2.empty());

        for (auto f_it = reg->template begin<RegionBase::NonTrivial>();
             f_it != reg->template end<RegionBase::NonTrivial>();
             ++f_it)
        {
          check(s3.count(*f_it));
          s3.erase(*f_it);
        }
        check(s3.empty());
      }

      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
      check(live_count == 0);
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