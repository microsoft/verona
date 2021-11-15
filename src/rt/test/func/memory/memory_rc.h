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

  template<class... T>
  uintptr_t alloc_garbage_helper(Alloc& alloc, Object* o, uintptr_t count)
  {
    alloc_in_region<T...>(alloc, o);
    auto ds = Region::debug_size(o);
    uintptr_t new_count = sizeof...(T) + count;
    check(ds == new_count); // o + T...
    return new_count;
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
    // Allocate a lot of garbage.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      uintptr_t obj_count = 1;

      obj_count =
        alloc_garbage_helper<C, F, MC, MF, LC, LF, XC, XF>(alloc, o, obj_count);
      obj_count = alloc_garbage_helper<C, C, C, XF, XF, MC, LC, LF, F, F, XC>(
        alloc, o, obj_count);
      obj_count = alloc_garbage_helper<C, C, C, XF, XF, MC, LC, LF, F, F, XC>(
        alloc, o, obj_count);
      obj_count = alloc_garbage_helper<XC, XC, XC, XF, MC>(alloc, o, obj_count);
      obj_count =
        alloc_garbage_helper<F, F, F, C, C, C, C, F>(alloc, o, obj_count);

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // Allocate a lot of objects that are all connected.
    // Then decref the root and see if children are deallocated.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      auto* o1 = new (alloc, o) C;
      auto* o2 = new (alloc, o) C;
      auto* o3 = new (alloc, o) C;
      auto* o4 = new (alloc, o) C;
      auto* o5 = new (alloc, o) C;
      auto* o6 = new (alloc, o) C;
      auto* o7 = new (alloc, o) C;

      // Link them up
      o1->f1 = o2;
      o1->f2 = o4;
      o2->f1 = o3;
      o2->f2 = o4;
      RegionRc::incref(o4, o);

      o4->f1 = o5;
      RegionRc::incref(o5, o);
      o5->f1 = o6;
      o5->f2 = o7;

      check(Region::debug_size(o) == 8);
      check(RegionRc::get_ref_count(o1, o) == 1);
      check(RegionRc::get_ref_count(o2, o) == 1);
      check(RegionRc::get_ref_count(o3, o) == 1);
      check(RegionRc::get_ref_count(o4, o) == 2);
      check(RegionRc::get_ref_count(o5, o) == 2);
      check(RegionRc::get_ref_count(o6, o) == 1);
      check(RegionRc::get_ref_count(o7, o) == 1);

      // Decref'ing o1 to 0 should trigger a deallocation.
      RegionRc::decref(alloc, o1, o);

      check(Region::debug_size(o) == 4);

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // Allocate objects which link to subregions and check that decrefing ISOs
    // has the correct behaviour.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      auto* sub1 = new (alloc) C;
      auto* sub2 = new (alloc) C;

      auto* o1 = new (alloc, o) C;
      auto* o2 = new (alloc, o) C;

      // Link them up
      o1->f1 = o2;
      o1->f2 = o;
      RegionRc::incref(o, o);
      o2->f1 = sub1;
      o2->f2 = sub2;

      check(Region::debug_size(o) == 3);
      check(RegionRc::get_ref_count(o, o) == 2);
      check(RegionRc::get_ref_count(o1, o) == 1);
      check(RegionRc::get_ref_count(o2, o) == 1);

      RegionRc::decref(alloc, o1, o);

      check(RegionRc::get_ref_count(o, o) == 1);

      check(Region::debug_size(o) == 1);

      Region::release(alloc, o);
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

      RegionRc::incref(o1, o);
      o6->f1 = o1;

      check(Region::debug_size(o) == 7);
      RegionRc::gc_cycles(alloc, o);
      check(Region::debug_size(o) == 7);

      o6->f1 = nullptr;
      RegionRc::decref(alloc, o1, o);

      RegionRc::gc_cycles(alloc, o);
      check(Region::debug_size(o) == 2);

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // A cycle where part of the subgraph is rooted by an external reference
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

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

      RegionRc::incref(o1, o);
      o6->f1 = o3;
      RegionRc::incref(o3, o);

      check(Region::debug_size(o) == 7);

      RegionRc::decref(alloc, o1, o);

      RegionRc::gc_cycles(alloc, o);
      check(Region::debug_size(o) == 7);

      // Now add a reference from o1->o6 and try and reclaim the cycle again.
      // cycle: (o1 -> o2 -> o3 -> o4 -> o5 -> o1)
      //          |                 ^
      //          +---------------> o6
      o1->f2 = o6;

      // Retrigger adding o1 to the lins stack.
      RegionRc::incref(o1, o);
      RegionRc::decref(alloc, o1, o);

      RegionRc::gc_cycles(alloc, o);
      check(Region::debug_size(o) == 1);

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // A cycle which contains references to other subregions.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      // Allocate some reachable objects.
      auto* o1 = new (alloc, o) C;
      auto* o2 = new (alloc, o) C;
      auto* o3 = new (alloc, o) C;

      // New subregion
      auto* p = new (alloc) C;
      auto* p1 = new (alloc, p) C;
      auto* p2 = new (alloc, p) C;

      p->f1 = p1;
      p1->f1 = p2;

      o1->f1 = o2;
      o2->f1 = o3;
      o2->f2 = p;
      o3->f1 = o1;

      RegionRc::incref(o1, o);
      RegionRc::decref(alloc, o1, o);

      check(Region::debug_size(o) == 4);
      check(Region::debug_size(p) == 3);

      RegionRc::gc_cycles(alloc, o);
      check(Region::debug_size(o) == 1);
      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }
    // A cycle where the root is the region's ISO.
    // This should tease out any bugs for special casing that is needed for
    // entry point objects.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      // Allocate some reachable objects.
      auto* o1 = new (alloc, o) C;
      auto* o2 = new (alloc, o) C;

      // cycle: (o -> o1 -> o2 -> o)
      o->f1 = o1;
      o1->f1 = o2;
      o2->f1 = o;

      RegionRc::incref(o, o);

      check(Region::debug_size(o) == 3);
      RegionRc::gc_cycles(alloc, o);
      check(Region::debug_size(o) == 3);

      RegionRc::incref(o, o);
      RegionRc::decref(alloc, o, o);

      RegionRc::gc_cycles(alloc, o);
      check(Region::debug_size(o) == 3);
      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }
  }

  void run_test()
  {
    test_basic();
    test_cycles();
  }
}
