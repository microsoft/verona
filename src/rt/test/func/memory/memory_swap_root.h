// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_swap_root
{
  /**
   * Helper for swap root tests
   * 1. Create a region with entry point type T
   * 2. Allocates objects of type Others,
   * 3. Makes the `index` element of Others the new root.
   * 4. Checks a few assertions.
   * 5. Adds a few more allocations
   * 6. Deallocates the region
   * 7. Checks a few more assertions.
   **/
  template<RegionType region_type, typename T, size_t index, typename... Others>
  void test_swap_root_helper()
  {
    auto oroot = new (region_type) T;
    Object* nroot;
    {
      UsingRegion r(oroot);

      nroot = allocs<index, Others...>();

      auto reg = Region::get(oroot);
      UNUSED(reg);

      set_entry_point(nroot);

      check(
        !oroot->debug_is_iso() && nroot->debug_is_iso() &&
        reg == Region::get(nroot));

      // Allocate a few more things to ensure our pointers are correct.
      new LargeC2;
      new LargeF2;
      new XLargeC2;
    }

    region_release(nroot);
    snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
    check(live_count == 0);
  }

  /**
   * Tests swapping the root (iso) object of a region.
   **/
  template<RegionType region_type>
  void test_swap_root()
  {
    using C = C1;
    using F = F1;
    using LC = LargeC2;
    using XC = XLargeC2;

    // Quick sanity check that swaps roots, allocates a few more objects,
    // merges two regions, and then allocates some more. We're checking that
    // region internal structures are still sensible.
    {
      F* nroot1 = nullptr;
      F* nroot2 = nullptr;
      auto oroot1 = new (region_type) C;
      {
        UsingRegion r(oroot1);

        nroot1 = new F;
        new XC;

        auto oroot2 = new (region_type) XC;
        {
          UsingRegion r(oroot2);
          nroot2 = allocs<0, F, F>();
          set_entry_point(nroot2);
        }

        set_entry_point(nroot1);

        check(!oroot1->debug_is_iso() && nroot1->debug_is_iso());
        check(!oroot2->debug_is_iso() && nroot2->debug_is_iso());
      }

      alloc_in_region<0, F, F, XC>(nroot1);

      alloc_in_region<0, C, C>(nroot2);

      {
        UsingRegion r(nroot1);
        merge(nroot2);
      }

      check(nroot1->debug_is_iso() && !nroot2->debug_is_iso());

      alloc_in_region<0, XC, XC, C>(nroot1);

      region_release(nroot1);
      snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
      check(live_count == 0);
    }

    if constexpr (region_type == RegionType::Trace)
    {
      // Swap two objects in the same ring. It doesn't matter if the object
      // needs finalisation, since the only thing that matters is primary vs
      // secondary ring, and if the iso object is always in the primary ring.

      // Only two objects in the ring, so we're swapping first with last.
      test_swap_root_helper<region_type, C, 0, C>();

      // More than two objects in the ring.
      // New root is right before the old root.
      test_swap_root_helper<region_type, C, 0, C, C, C>();

      // More than two objects in the ring.
      // New root is somewhere in the middle of the ring.
      test_swap_root_helper<region_type, C, 1, C, C, C>();

      // More than two objects in the ring.
      // New root is right after the region metadata object.
      test_swap_root_helper<region_type, C, 1, C, C, C>();

      // Swap objects from different rings. It doesn't matter which object needs
      // finalisation, since we are swapping the iso object (from the primary
      // ring) with an object in the secondary ring.

      // One object in each ring.
      test_swap_root_helper<region_type, F, 0, C>();

      // Two objects in primary ring, multiple in secondary ring.
      // New root is the same as the new "old root" (after swapping rings).
      test_swap_root_helper<region_type, F, 1, F, C, C, C>();

      // Two objects in primary ring, multiple in secondary ring.
      // New root is right before the new "old root" (after swapping rings).
      test_swap_root_helper<region_type, F, 2, F, C, C, C>();

      // Two objects in primary ring, multiple in secondary ring.
      // New root is somewhere in middle of ring.
      test_swap_root_helper<region_type, F, 3, F, C, C, C, C, C>();

      // Two objects in primary ring, multiple in secondary ring.
      // New root is right after the region metadata object.
      test_swap_root_helper<region_type, F, 3, F, C, C, C>();
    }
    else if constexpr (region_type == RegionType::Arena)
    {
      // Both objects in the same arena.
      test_swap_root_helper<region_type, C, 0, C>();

      // Objects in different arenas.
      test_swap_root_helper<region_type, C, 3, LC, LC, LC, C>();

      // Old root in the ring, new root in an arena.
      test_swap_root_helper<region_type, XC, 0, C>();

      // Old root in an arena, new root in the ring.
      // Only one object in the ring.
      test_swap_root_helper<region_type, C, 0, XC>();

      // Old root in an arena, new root in the ring.
      // Two objects in the ring. New root is last in the ring.
      test_swap_root_helper<region_type, C, 0, XC, XC>();

      // Old root in an arena, new root in the ring.
      // Two objects in the ring. New root is first in the ring.
      test_swap_root_helper<region_type, C, 1, XC, XC>();

      // Old root in an arena, new root in the ring.
      // Multiple objects in the ring. New root is last in the ring.
      test_swap_root_helper<region_type, C, 0, XC, XC, XC>();

      // Old root in an arena, new root in the ring.
      // Multiple objects in the ring. New root is in the middle of the ring.
      test_swap_root_helper<region_type, C, 1, XC, XC, XC>();

      // Old root in an arena, new root in the ring.
      // Multiple objects in the ring. New root is first in the ring.
      test_swap_root_helper<region_type, C, 2, XC, XC, XC>();

      // Both objects in the ring. Old root must be last object in the ring.
      // Only two objects in the ring.
      test_swap_root_helper<region_type, XC, 0, XC>();

      // Both objects in the ring. Old root must be last object in the ring.
      // Multiple objects in the ring. New root is right before the old root.
      test_swap_root_helper<region_type, XC, 0, XC, XC, XC>();

      // Both objects in the ring. Old root must be last object in the ring.
      // Multiple objects in the ring. New root is somewhere in the middle.
      test_swap_root_helper<region_type, XC, 1, XC, XC, XC>();

      // Both objects in the ring. Old root must be last object in the ring.
      // Multiple objects in the ring. New root is after region metadata object.
      test_swap_root_helper<region_type, XC, 2, XC, XC, XC>();
    }
  }

  void run_test()
  {
    test_swap_root<RegionType::Trace>();
    test_swap_root<RegionType::Arena>();
  }
}