// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_alloc
{
  /**
   * Helper for `test_alloc` that allocates a sequence of objects in a region,
   * and then frees that region.
   *
   * E.g. calling `test_alloc_helper<A, B, C>()` will allocate 3 objects, in the
   * order A, B, and C. A will be the type of the iso object.
   **/
  template<RegionType region_type, class First, class... Rest>
  void test_alloc_helper()
  {
    auto a = alloc_region<First, Rest...>(region_type);
    region_release(a);
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    check(live_count == 0);
  }

  /**
   * Tests allocating various sequences of objects within a region.
   **/
  template<RegionType region_type>
  void test_alloc()
  {
    using C = C1;
    using F = F1;
    using MC = MediumC2;
    using MF = MediumF2;
    using LC = LargeC2;
    using LF = LargeF2;
    using XC = XLargeC2;
    using XF = XLargeF2;

    if constexpr (region_type == RegionType::Trace)
    {
      // Region contains only the iso object.
      test_alloc_helper<region_type, C>();

      // Three objects in the same ring.
      test_alloc_helper<region_type, C, C, C>();
      test_alloc_helper<region_type, F, F, F>();

      // Primary ring has no finalisers, secondary ring needs finalisers.
      test_alloc_helper<region_type, C, C, F, F>();

      // Primary ring needs finalisers, secondary ring has no finalisers.
      test_alloc_helper<region_type, F, F, C, C>();
    }
    else if constexpr (region_type == RegionType::Arena)
    {
      // Region contains only the iso object. Objects needing and not needing
      // finalisers are allocated differently.
      test_alloc_helper<region_type, C>();
      test_alloc_helper<region_type, F>();

      // Region contains only the iso object, which just fis into an arena.
      test_alloc_helper<region_type, LC>();
      test_alloc_helper<region_type, LF>();

      // Region contains only the iso object, which is in the large object ring.
      test_alloc_helper<region_type, XC>();
      test_alloc_helper<region_type, XF>();

      // Try different sizes of objects, to test padding/rounding.
      using S1 = C2<32>;
      using S2 = C2<36>;
      using S3 = C2<40>;
      using S4 = C2<44>;
      using S5 = C2<48>;
      using S6 = C2<52>;
      using S7 = C2<56>;
      check(
        sizeof(S1) % Object::ALIGNMENT != 0 ||
        sizeof(S2) % Object::ALIGNMENT != 0 ||
        sizeof(S3) % Object::ALIGNMENT != 0 ||
        sizeof(S4) % Object::ALIGNMENT != 0 ||
        sizeof(S5) % Object::ALIGNMENT != 0 ||
        sizeof(S6) % Object::ALIGNMENT != 0 ||
        sizeof(S7) % Object::ALIGNMENT != 0);
      test_alloc_helper<region_type, S1, S1, S1>();
      test_alloc_helper<region_type, S2, S2, S2>();
      test_alloc_helper<region_type, S3, S3, S3>();
      test_alloc_helper<region_type, S4, S4, S4>();
      test_alloc_helper<region_type, S5, S5, S5>();
      test_alloc_helper<region_type, S6, S6, S6>();
      test_alloc_helper<region_type, S7, S7, S7>();

      // For an arena region, there are essentially three "positions" an object
      // can be in: the non-finalisers section of an arena, the finalisers
      // section of an arena, or the large object ring. We want to test putting
      // the iso object and other objects in all of those positions, and we also
      // want multiple objects in those positions. Finally, we want to test
      // "filling" up an arena and causing a new one to be allocated.
      //
      // There are far too many combinations to test all of them, and some
      // combinations are effectively redundant. The following selection should
      // hopefully cover enough combinations.

      // Small objects in arena.
      test_alloc_helper<region_type, C, C, C>();
      test_alloc_helper<region_type, F, F, F>();
      test_alloc_helper<region_type, C, F, C, F>();

      // Medium objects that need a new arena to be allocated. Only 2 can fit in
      // a single arena, so we'll need a second arena.
      test_alloc_helper<region_type, MC, MC, MC>();
      test_alloc_helper<region_type, MF, MF, MF>();
      test_alloc_helper<region_type, MF, MC, MF>();
      test_alloc_helper<region_type, MC, MF, MC>();

      // Three large objects in the ring. It doesn't matter what kind of objects
      // they are.
      test_alloc_helper<region_type, XC, XF, XC>();

      // Two large objects in ring, including iso. One small object in arena.
      test_alloc_helper<region_type, XC, XC, C>();
      test_alloc_helper<region_type, XF, XF, F>();

      // Large iso object in ring. Two small objects in arena.
      test_alloc_helper<region_type, XC, C, C>();
      test_alloc_helper<region_type, XF, F, F>();

      // Iso object in arena. Two large objects in ring.
      test_alloc_helper<region_type, C, XF, XF>();
      test_alloc_helper<region_type, F, XC, XC>();

      // Many (8) objects in the large object ring.
      test_alloc_helper<region_type, XC, XC, XC, XC, XF, XF, XF, XF>();

      // Many (8) object arenas, each with a single object.
      test_alloc_helper<region_type, LC, LC, LC, LC, LF, LF, LF, LF>();

      // Many (8) objects in an arena.
      test_alloc_helper<region_type, C, C, C, C, F, F, F, F>();

      // Many objects in many arenas.
      // 1st arena: MC MF C F
      // 2nd arena: MF MC F F
      // 3rd arena: MC MC C C
      test_alloc_helper<
        region_type,
        MC,
        MF,
        C,
        F,
        MF,
        MC,
        F,
        F,
        MC,
        MC,
        C,
        C>();
    }
    else if constexpr (region_type == RegionType::Rc)
    {
      // Region contains only the iso object.
      test_alloc_helper<region_type, C>();

      test_alloc_helper<region_type, C, C, C>();
      test_alloc_helper<region_type, F, F, F>();
      test_alloc_helper<region_type, C, C, F, F>();
      test_alloc_helper<region_type, F, F, C, C>();
    }
  }

  void run_test()
  {
    test_alloc<RegionType::Trace>();
    test_alloc<RegionType::Arena>();
  }
}