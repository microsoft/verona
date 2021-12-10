// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_rc
{
  constexpr auto region_type = RegionType::Rc;
  using C = C1<region_type>;
  using F = F1<region_type>;
  using MC = MediumC2<region_type>;
  using MF = MediumF2<region_type>;
  using LC = LargeC2<region_type>;
  using LF = LargeF2<region_type>;
  using XC = XLargeC2<region_type>;
  using XF = XLargeF2<region_type>;

  using Cx = C3<region_type>;
  using Fx = F3<region_type>;

  void push_lins_stack(Alloc& alloc, Object* o, Object* in)
  {
    RegionRc::incref(o);
    RegionRc::decref(alloc, o, in);
  }

  /**
   * A few basic tests to start:
   *   - allocating unreachable objects, and then releasing the region to ensure
   *   they are all deallocated.
   *   - allocating objects and making them all reachable, but then decrefing
   *   objects to 0 to make sure that deallocation is recursive.
   **/
  void test_basic()
  {
    // Allocate a lot of objects that are all connected.
    // Then decref the root and see if children are deallocated.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      {
        UsingRegion rc(o);

        auto* o1 = new (alloc, o) C;
        auto* o2 = new (alloc, o) C;
        auto* o3 = new (alloc, o) C;
        auto* o4 = new (alloc, o) C;
        auto* o5 = new (alloc, o) C;
        auto* o6 = new (alloc, o) C;
        auto* o7 = new (alloc, o) C;

        // Link them up
        o->f1 = o1;
        o1->f1 = o2;
        o1->f2 = o4;
        o2->f1 = o3;
        o2->f2 = o4;
        RegionRc::incref(o4);

        o4->f1 = o5;
        o5->f1 = o6;
        o5->f2 = o7;

        check(rc.debug_size() == 8);
        check(rc.debug_get_ref_count(o1) == 1);
        check(rc.debug_get_ref_count(o2) == 1);
        check(rc.debug_get_ref_count(o3) == 1);
        check(rc.debug_get_ref_count(o4) == 2);
        check(rc.debug_get_ref_count(o5) == 1);
        check(rc.debug_get_ref_count(o6) == 1);
        check(rc.debug_get_ref_count(o7) == 1);

        // Decref'ing o5 to 0 should trigger a deallocation.
        o4->f1 = nullptr;
        RegionRc::decref(alloc, o5, o);

        check(rc.debug_size() == 5);
        Region::release(alloc, o);
      }
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // Allocate objects which link to subregions and check that decrefing ISOs
    // has the correct behaviour.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      auto* sub1 = new (alloc) C;
      auto* sub2 = new (alloc) C;

      {
        UsingRegion rc(o);
        auto* o1 = new (alloc, o) C;
        auto* o2 = new (alloc, o) C;

        // Link them up
        o->f1 = o1;
        o1->f1 = o2;
        o1->f2 = o;
        RegionRc::incref(o);
        o2->f1 = sub1;
        o2->f2 = sub2;

        check(rc.debug_size() == 3);
        check(rc.debug_get_ref_count(o) == 2);
        check(rc.debug_get_ref_count(o1) == 1);
        check(rc.debug_get_ref_count(o2) == 1);

        o->f1 = nullptr;
        RegionRc::decref(alloc, o1, o);

        check(rc.debug_get_ref_count(o) == 1);

        check(rc.debug_size() == 1);
        Region::release(alloc, o);
      }
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }
  }

  /**
   * Show that basic reference counting can't handle cycles.
   **/
  void test_cycles()
  {
    // Start with a simple cycle
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      {
        UsingRegion rc(o);

        // Allocate some reachable objects.
        auto* o1 = new (alloc, o) C;
        auto* o2 = new (alloc, o) C;
        auto* o3 = new (alloc, o) C;
        auto* o4 = new (alloc, o) C;
        auto* o5 = new (alloc, o) C;
        auto* o6 = new (alloc, o) C;

        // cycle: o6 -> (o1 -> o2 -> o3 -> o4 -> o5 -> o1)
        o1->f1 = o2;
        o2->f1 = o3;
        o3->f1 = o4;
        o4->f1 = o5;
        o5->f1 = o1;

        RegionRc::incref(o1);
        o6->f1 = o1;

        check(rc.debug_size() == 7);
        RegionRc::gc_cycles(alloc, o);
        check(rc.debug_size() == 7);

        o6->f1 = nullptr;
        RegionRc::decref(alloc, o1, o);

        RegionRc::gc_cycles(alloc, o);
        check(rc.debug_size() == 2);

        // Re-link the object graph so that release doesn't leave anything.
        o->f1 = o6;
        Region::release(alloc, o);
      }
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // A cycle where part of the subgraph is rooted by an external reference
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      {
        UsingRegion rc(o);

        // Allocate some reachable objects.
        auto* o1 = new (alloc, o) C;
        auto* o2 = new (alloc, o) C;
        auto* o3 = new (alloc, o) C;
        auto* o4 = new (alloc, o) C;
        auto* o5 = new (alloc, o) C;
        auto* o6 = new (alloc, o) C;

        // cycle: (o1 -> o2 -> o3 -> o4 -> o5 -> o1)
        //                     ^
        //                     o6
        o1->f1 = o2;
        o2->f1 = o3;
        o3->f1 = o4;
        o4->f1 = o5;
        o5->f1 = o1;

        RegionRc::incref(o1);
        o6->f1 = o3;
        RegionRc::incref(o3);

        check(rc.debug_size() == 7);

        RegionRc::decref(alloc, o1, o);

        RegionRc::gc_cycles(alloc, o);
        check(rc.debug_size() == 7);

        // Now add a reference from o1->o6 and try and reclaim the cycle again.
        // cycle: (o1 -> o2 -> o3 -> o4 -> o5 -> o1)
        //          |                 ^
        //          +---------------> o6
        o1->f2 = o6;

        // Retrigger adding o1 to the lins stack.
        RegionRc::incref(o1);
        RegionRc::decref(alloc, o1, o);

        RegionRc::gc_cycles(alloc, o);
        check(rc.debug_size() == 1);
        Region::release(alloc, o);
      }

      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // A cycle which contains references to other subregions.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      // New subregion
      auto* p = new (alloc) C;
      {
        UsingRegion rc(p);
        auto* p1 = new (alloc, p) C;
        auto* p2 = new (alloc, p) C;

        p->f1 = p1;
        p1->f1 = p2;
      }

      {
        UsingRegion rc(o);

        // Allocate some reachable objects.
        auto* o1 = new (alloc, o) C;
        auto* o2 = new (alloc, o) C;
        auto* o3 = new (alloc, o) C;

        o1->f1 = o2;
        o2->f1 = o3;
        o2->f2 = p;
        o3->f1 = o1;

        RegionRc::incref(o1);
        RegionRc::decref(alloc, o1, o);

        check(rc.debug_size() == 4);

        RegionRc::gc_cycles(alloc, o);
        check(rc.debug_size() == 1);
        Region::release(alloc, o);
      }
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // A cycle where the root is the region's ISO.
    // This should tease out any bugs for special casing that is needed for
    // entry point objects.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;
      {
        UsingRegion rc(o);

        // Allocate some reachable objects.
        auto* o1 = new (alloc, o) C;
        auto* o2 = new (alloc, o) C;

        // cycle: (o -> o1 -> o2 -> o)
        o->f1 = o1;
        o1->f1 = o2;
        o2->f1 = o;
        RegionRc::incref(o);

        /* push_lins_stack(alloc, o, o); */

        /* check(rc.debug_size() == 3); */
        /* RegionRc::gc_cycles(alloc, o); */
        /* check(rc.debug_size() == 3); */

        o2->f1 = o1;
        RegionRc::decref(alloc, o, o);
        RegionRc::incref(o1);

        push_lins_stack(alloc, o1, o);
        Region::release(alloc, o);
      }
    }
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  void run_test()
  {
    test_basic();
    test_cycles();
  }
}
