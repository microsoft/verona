// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_swap_root
{
  /**
   * Helper for swap root tests that calls the actual swap_root method, then
   * allocates a few more objects to ensure the region pointers are set
   * properly, and then releases the regions and makes a few assertions.
   **/
  template<RegionType region_type>
  void test_swap_root_helper(Alloc& alloc, Object* oroot, Object* nroot)
  {
    using RegionClass = typename RegionType_to_class<region_type>::T;

    auto reg = Region::get(oroot);
    UNUSED(reg);

    RegionClass::swap_root(oroot, nroot);
    check(
      !oroot->debug_is_iso() && nroot->debug_is_iso() &&
      reg == Region::get(nroot));

    // Allocate a few more things to ensure our pointers are correct.
    new (alloc, nroot) LargeC2<region_type>;
    new (alloc, nroot) LargeF2<region_type>;
    new (alloc, nroot) XLargeC2<region_type>;

    Region::release(alloc, nroot);
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    check(live_count == 0);
  }

  /**
   * Tests swapping the root (iso) object of a region.
   **/
  template<RegionType region_type>
  void test_swap_root()
  {
    using RegionClass = typename RegionType_to_class<region_type>::T;
    using C = C1<region_type>;
    using F = F1<region_type>;
    using LC = LargeC2<region_type>;
    using XC = XLargeC2<region_type>;

    // Quick sanity check that swaps roots, allocates a few more objects,
    // merges two regions, and then allocates some more. We're checking that
    // region internal structures are still sensible.
    {
      auto& alloc = ThreadAlloc::get();

      C* oroot1 = new (alloc) C;
      F* nroot1 = new (alloc, oroot1) F;
      new (alloc, oroot1) XC;

      XC* oroot2 = new (alloc) XC;
      F* nroot2 = new (alloc, oroot2) F;
      new (alloc, oroot2) F;

      RegionClass::swap_root(oroot1, nroot1);
      RegionClass::swap_root(oroot2, nroot2);
      check(!oroot1->debug_is_iso() && nroot1->debug_is_iso());
      check(!oroot2->debug_is_iso() && nroot2->debug_is_iso());

      alloc_in_region<F, F, XC>(alloc, nroot1);
      alloc_in_region<C, C>(alloc, nroot2);

      RegionClass::merge(alloc, nroot1, nroot2);
      check(nroot1->debug_is_iso() && !nroot2->debug_is_iso());

      alloc_in_region<XC, XC, C>(alloc, nroot1);

      Region::release(alloc, nroot1);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
      check(live_count == 0);
    }

    if constexpr (region_type == RegionType::Trace)
    {
      // Swap two objects in the same ring. It doesn't matter if the object
      // needs finalisation, since the only thing that matters is primary vs
      // secondary ring, and if the iso object is always in the primary ring.

      // Only two objects in the ring, so we're swapping first with last.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        C* nroot = new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // More than two objects in the ring.
      // New root is right before the old root.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        C* nroot = new (alloc, oroot) C;
        new (alloc, oroot) C;
        new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // More than two objects in the ring.
      // New root is somewhere in the middle of the ring.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        new (alloc, oroot) C;
        C* nroot = new (alloc, oroot) C;
        new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // More than two objects in the ring.
      // New root is right after the region metadata object.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        new (alloc, oroot) C;
        new (alloc, oroot) C;
        C* nroot = new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Swap objects from different rings. It doesn't matter which object needs
      // finalisation, since we are swapping the iso object (from the primary
      // ring) with an object in the secondary ring.

      // One object in each ring.
      {
        auto& alloc = ThreadAlloc::get();
        F* oroot = new (alloc) F;
        C* nroot = new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Two objects in primary ring, multiple in secondary ring.
      // New root is the same as the new "old root" (after swapping rings).
      {
        auto& alloc = ThreadAlloc::get();
        F* oroot = new (alloc) F;
        new (alloc, oroot) F;
        C* nroot = new (alloc, oroot) C;
        new (alloc, oroot) C;
        new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Two objects in primary ring, multiple in secondary ring.
      // New root is right before the new "old root" (after swapping rings).
      {
        auto& alloc = ThreadAlloc::get();
        F* oroot = new (alloc) F;
        new (alloc, oroot) F;
        new (alloc, oroot) C;
        C* nroot = new (alloc, oroot) C;
        new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Two objects in primary ring, multiple in secondary ring.
      // New root is somewhere in middle of ring.
      {
        auto& alloc = ThreadAlloc::get();
        F* oroot = new (alloc) F;
        new (alloc, oroot) F;
        new (alloc, oroot) C;
        new (alloc, oroot) C;
        C* nroot = new (alloc, oroot) C;
        new (alloc, oroot) C;
        new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Two objects in primary ring, multiple in secondary ring.
      // New root is right after the region metadata object.
      {
        auto& alloc = ThreadAlloc::get();
        F* oroot = new (alloc) F;
        new (alloc, oroot) F;
        new (alloc, oroot) C;
        new (alloc, oroot) C;
        C* nroot = new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }
    }
    else if constexpr (region_type == RegionType::Arena)
    {
      // Both objects in the same arena.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        C* nroot = new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Objects in different arenas.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        new (alloc, oroot) LC;
        new (alloc, oroot) LC;
        new (alloc, oroot) LC;
        C* nroot = new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Old root in the ring, new root in an arena.
      {
        auto& alloc = ThreadAlloc::get();
        XC* oroot = new (alloc) XC;
        C* nroot = new (alloc, oroot) C;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Old root in an arena, new root in the ring.
      // Only one object in the ring.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        XC* nroot = new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Old root in an arena, new root in the ring.
      // Two objects in the ring. New root is last in the ring.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        XC* nroot = new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Old root in an arena, new root in the ring.
      // Two objects in the ring. New root is first in the ring.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        new (alloc, oroot) XC;
        XC* nroot = new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Old root in an arena, new root in the ring.
      // Multiple objects in the ring. New root is last in the ring.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        XC* nroot = new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Old root in an arena, new root in the ring.
      // Multiple objects in the ring. New root is in the middle of the ring.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        new (alloc, oroot) XC;
        XC* nroot = new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Old root in an arena, new root in the ring.
      // Multiple objects in the ring. New root is first in the ring.
      {
        auto& alloc = ThreadAlloc::get();
        C* oroot = new (alloc) C;
        new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        XC* nroot = new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Both objects in the ring. Old root must be last object in the ring.
      // Only two objects in the ring.
      {
        auto& alloc = ThreadAlloc::get();
        XC* oroot = new (alloc) XC;
        XC* nroot = new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Both objects in the ring. Old root must be last object in the ring.
      // Multiple objects in the ring. New root is right before the old root.
      {
        auto& alloc = ThreadAlloc::get();
        XC* oroot = new (alloc) XC;
        XC* nroot = new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Both objects in the ring. Old root must be last object in the ring.
      // Multiple objects in the ring. New root is somewhere in the middle.
      {
        auto& alloc = ThreadAlloc::get();
        XC* oroot = new (alloc) XC;
        new (alloc, oroot) XC;
        XC* nroot = new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }

      // Both objects in the ring. Old root must be last object in the ring.
      // Multiple objects in the ring. New root is after region metadata object.
      {
        auto& alloc = ThreadAlloc::get();
        XC* oroot = new (alloc) XC;
        new (alloc, oroot) XC;
        new (alloc, oroot) XC;
        XC* nroot = new (alloc, oroot) XC;
        test_swap_root_helper<region_type>(alloc, oroot, nroot);
      }
    }
  }

  void run_test()
  {
    test_swap_root<RegionType::Trace>();
    test_swap_root<RegionType::Arena>();
  }
}