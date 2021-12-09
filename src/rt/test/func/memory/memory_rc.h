// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_rc
{
  constexpr auto region_type = RegionType::Rc;
  using C = C1;
  using F = F1;
  using MC = MediumC2;
  using MF = MediumF2;
  using LC = LargeC2;
  using LF = LargeF2;
  using XC = XLargeC2;
  using XF = XLargeF2;

  using Cx = C3;
  using Fx = F3;

  void push_lins_stack(Object* o)
  {
    incref(o);
    decref(o);
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
      auto* o = new (RegionType::Rc) C;

      {
        UsingRegion rc(o);

        auto* o1 = new C;
        auto* o2 = new C;
        auto* o3 = new C;
        auto* o4 = new C;
        auto* o5 = new C;
        auto* o6 = new C;
        auto* o7 = new C;

        // Link them up
        o->f1 = o1;
        o1->f1 = o2;
        o1->f2 = o4;
        o2->f1 = o3;
        o2->f2 = o4;
        incref(o4);

        o4->f1 = o5;
        o5->f1 = o6;
        o5->f2 = o7;

        check(debug_size() == 8);
        check(debug_get_ref_count(o1) == 1);
        check(debug_get_ref_count(o2) == 1);
        check(debug_get_ref_count(o3) == 1);
        check(debug_get_ref_count(o4) == 2);
        check(debug_get_ref_count(o5) == 1);
        check(debug_get_ref_count(o6) == 1);
        check(debug_get_ref_count(o7) == 1);

        // Decref'ing o5 to 0 should trigger a deallocation.
        o4->f1 = nullptr;
        decref(o5);

        check(debug_size() == 5);
      }
      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // Allocate objects which link to subregions and check that decrefing ISOs
    // has the correct behaviour.
    {
      auto* o = new (RegionType::Rc) C;

      auto* sub1 = new (RegionType::Rc) C;
      auto* sub2 = new (RegionType::Rc) C;

      {
        UsingRegion rc(o);
        auto* o1 = new C;
        auto* o2 = new C;

        // Link them up
        o->f1 = o1;
        o1->f1 = o2;
        o1->f2 = o;
        incref(o);
        o2->f1 = sub1;
        o2->f2 = sub2;

        check(debug_size() == 3);
        check(debug_get_ref_count(o) == 2);
        check(debug_get_ref_count(o1) == 1);
        check(debug_get_ref_count(o2) == 1);

        o->f1 = nullptr;
        decref(o1);

        check(debug_get_ref_count(o) == 1);

        check(debug_size() == 1);
      }
      region_release(o);
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
      auto* o = new (RegionType::Rc) C;

      {
        UsingRegion rc(o);

        // Allocate some reachable objects.
        auto* o1 = new C;
        auto* o2 = new C;
        auto* o3 = new C;
        auto* o4 = new C;
        auto* o5 = new C;
        auto* o6 = new C;

        // cycle: o6 -> (o1 -> o2 -> o3 -> o4 -> o5 -> o1)
        o1->f1 = o2;
        o2->f1 = o3;
        o3->f1 = o4;
        o4->f1 = o5;
        o5->f1 = o1;

        incref(o1);
        o6->f1 = o1;

        check(debug_size() == 7);
        region_collect();
        check(debug_size() == 7);

        o6->f1 = nullptr;
        decref(o1);

        region_collect();
        check(debug_size() == 2);

        // Re-link the object graph so that release doesn't leave anything.
        o->f1 = o6;
      }
      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // A cycle where part of the subgraph is rooted by an external reference
    {
      auto* o = new (RegionType::Rc) C;

      {
        UsingRegion rc(o);

        // Allocate some reachable objects.
        auto* o1 = new C;
        auto* o2 = new C;
        auto* o3 = new C;
        auto* o4 = new C;
        auto* o5 = new C;
        auto* o6 = new C;

        // cycle: (o1 -> o2 -> o3 -> o4 -> o5 -> o1)
        //                     ^
        //                     o6
        o1->f1 = o2;
        o2->f1 = o3;
        o3->f1 = o4;
        o4->f1 = o5;
        o5->f1 = o1;

        incref(o1);
        o6->f1 = o3;
        incref(o3);

        check(debug_size() == 7);

        decref(o1);

        region_collect();
        check(debug_size() == 7);

        // Now add a reference from o1->o6 and try and reclaim the cycle again.
        // cycle: (o1 -> o2 -> o3 -> o4 -> o5 -> o1)
        //          |                 ^
        //          +---------------> o6
        o1->f2 = o6;

        // Retrigger adding o1 to the lins stack.
        incref(o1);
        decref(o1);

        region_collect();
        check(debug_size() == 1);
      }
      region_release(o);

      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // A cycle which contains references to other subregions.
    {
      auto* o = new (RegionType::Rc) C;

      // New subregion
      auto* p = new (RegionType::Rc) C;
      {
        UsingRegion rc(p);
        auto* p1 = new C;
        auto* p2 = new C;

        p->f1 = p1;
        p1->f1 = p2;
      }

      {
        UsingRegion rc(o);

        // Allocate some reachable objects.
        auto* o1 = new C;
        auto* o2 = new C;
        auto* o3 = new C;

        o1->f1 = o2;
        o2->f1 = o3;
        o2->f2 = p;
        o3->f1 = o1;

        incref(o1);
        decref(o1);

        check(debug_size() == 4);

        region_collect();
        check(debug_size() == 1);
      }
      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // A cycle where the root is the region's ISO.
    // This should tease out any bugs for special casing that is needed for
    // entry point objects.
    {
      auto* o = new (RegionType::Rc) C;
      {
        UsingRegion rc(o);

        // Allocate some reachable objects.
        auto* o1 = new C;
        auto* o2 = new C;

        // cycle: (o -> o1 -> o2 -> o)
        o->f1 = o1;
        o1->f1 = o2;
        o2->f1 = o;
        incref(o);

        /* push_lins_stack(o); */

        /* check(debug_size() == 3); */
        /* region_collect(); */
        /* check(debug_size() == 3); */

        o2->f1 = o1;
        decref(o);
        incref(o1);

        push_lins_stack(o1);
      }
      region_release(o);
    }
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  void run_test()
  {
    test_basic();
    test_cycles();
  }
}
