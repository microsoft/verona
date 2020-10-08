// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_merge
{
  /**
   * Helper for merge tests that calls the actual merge method, then allocates
   * a few more objects to ensure the region pointers are set properly, and then
   * releases the regions and makes a few assertions.
   **/
  template<RegionType region_type>
  void test_merge_helper(Alloc* alloc, Object* r1, Object* r2)
  {
    using RegionClass = typename RegionType_to_class<region_type>::T;

    RegionClass::merge(alloc, r1, r2);
    check(!r2->debug_is_iso());

    // Allocate a few more things to ensure our pointers are correct.
    new (alloc, r1) LargeC2<region_type>;
    new (alloc, r1) LargeF2<region_type>;
    new (alloc, r1) XLargeC2<region_type>;

    Region::release(alloc, r1);
    snmalloc::current_alloc_pool()->debug_check_empty();
    check(live_count == 0);
  }

  /**
   * Tests merging two regions.
   **/
  template<RegionType region_type>
  void test_merge()
  {
    using RegionClass = typename RegionType_to_class<region_type>::T;
    using C = C1<region_type>;
    using F = F1<region_type>;
    using LC = LargeC2<region_type>;
    using XC = XLargeC2<region_type>;

    // Quick sanity check that merges two regions, allocates a few more
    // objects, swaps roots a few times, and allocates more objects. We're
    // checking that region internal structures are still sensible.
    {
      auto* alloc = ThreadAlloc::get();

      C* r1 = new (alloc) C;
      new (alloc, r1) F;
      XC* o1 = new (alloc, r1) XC;

      F* r2 = new (alloc) F;
      new (alloc, r2) F;
      C* o2 = new (alloc, r2) C;
      new (alloc, r2) LC;

      RegionClass::merge(alloc, r1, r2);
      check(r1->debug_is_iso() && !r2->debug_is_iso());

      alloc_in_region<F, F, LC, XC>(alloc, r1);

      RegionClass::swap_root(r1, o1);
      alloc_in_region<C, C>(alloc, o1);
      RegionClass::swap_root(o1, r2);
      alloc_in_region<LC, F>(alloc, r2);
      RegionClass::swap_root(r2, o2);
      check(o2->debug_is_iso());

      alloc_in_region<F, LC>(alloc, o2);

      Region::release(alloc, o2);
      snmalloc::current_alloc_pool()->debug_check_empty();
      check(live_count == 0);
    }

    if constexpr (region_type == RegionType::Trace)
    {
      // There are 72 potential test cases. The primary ring can be
      // {singleton, multiple} and the secondary ring can be
      // {empty, singleton, multiple}. So there are 6 different "shapes" for a
      // region, and 36 "shapes" for merging two regions. Furthermore, the
      // primary rings of the two regions may or may not match, which brings us
      // to 72. Testing all of them would be far too tedious, so we test a
      // selection.

      // For the secondary ring, we'll combine {singleton, multiple}, which
      // gives us 4 shapes for a region, and 16 shapes to merge. For half of
      // those, we'll make the primary rings mismatch.

      // Primary ring:   {singleton x singleton} (mismatch)
      // Secondary ring: {empty x empty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C>(alloc);
        auto* r2 = alloc_region<F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {singleton x singleton} (mismatch)
      // Secondary ring: {empty x nonempty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C>(alloc);
        auto* r2 = alloc_region<F, C, C>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {singleton x singleton} (mismatch)
      // Secondary ring: {nonempty x empty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, F, F>(alloc);
        auto* r2 = alloc_region<F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {singleton x singleton} (mismatch)
      // Secondary ring: {nonempty x nonempty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, F, F>(alloc);
        auto* r2 = alloc_region<F, C, C>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {singleton x multiple}
      // Secondary ring: {empty x empty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C>(alloc);
        auto* r2 = alloc_region<C, C, C>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {singleton x multiple}
      // Secondary ring: {empty x nonempty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C>(alloc);
        auto* r2 = alloc_region<C, C, F, F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {singleton x multiple}
      // Secondary ring: {nonempty x empty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<F, C, C>(alloc);
        auto* r2 = alloc_region<F, F, F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {singleton x multiple}
      // Secondary ring: {nonempty x nonempty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, F, F>(alloc);
        auto* r2 = alloc_region<C, C, F, F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {multiple x singleton}
      // Secondary ring: {empty x empty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, C, C>(alloc);
        auto* r2 = alloc_region<C>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {multiple x singleton}
      // Secondary ring: {empty x nonempty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, C, C>(alloc);
        auto* r2 = alloc_region<C, F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {multiple x singleton}
      // Secondary ring: {nonempty x empty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<F, F, C>(alloc);
        auto* r2 = alloc_region<F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {multiple x singleton}
      // Secondary ring: {nonempty x nonempty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, C, F>(alloc);
        auto* r2 = alloc_region<C, F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {multiple x multiple} (mismatch)
      // Secondary ring: {empty x empty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, C, C>(alloc);
        auto* r2 = alloc_region<F, F, F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {multiple x multiple} (mismatch)
      // Secondary ring: {empty x nonempty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<F, F, F>(alloc);
        auto* r2 = alloc_region<C, C, F>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {multiple x multiple} (mismatch)
      // Secondary ring: {nonempty x empty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<F, F, F, C>(alloc);
        auto* r2 = alloc_region<C, C>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Primary ring:   {multiple x multiple} (mismatch)
      // Secondary ring: {nonempty x nonempty}
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, C, C, F, F>(alloc);
        auto* r2 = alloc_region<F, F, F, C, C>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }
    }
    else if constexpr (region_type == RegionType::Arena)
    {
      // Nine cases for merging two arena linked lists or two object rings.
      // {empty, singleton, multiple} x {empty, singleton, multiple}
      //
      // Note that cases overlap, e.g. if both arena linked lists are empty,
      // then both large object rings must be non-empty.

      // Arena linked lists: empty x empty
      // Large object rings: multiple x multiple
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<XC, XC>(alloc);
        auto* r2 = alloc_region<XC, XC>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Arena linked lists: empty x singleton
      // Large object rings: multiple x singleton
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<XC, XC>(alloc);
        auto* r2 = alloc_region<XC, C>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Arena linked lists: empty x multiple
      // Large object rings: multiple x empty
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<XC, XC>(alloc);
        auto* r2 = alloc_region<LC, LC>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Arena linked lists: singleton x empty
      // Large object rings: singleton x multiple
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, XC>(alloc);
        auto* r2 = alloc_region<XC, XC>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Arena linked lists: singleton x singleton
      // Large object rings: singleton x singleton
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, XC>(alloc);
        auto* r2 = alloc_region<C, XC>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Arena linked lists: singleton x multiple
      // Large object rings: singleton x empty
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<C, XC>(alloc);
        auto* r2 = alloc_region<LC, LC>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Arena linked lists: multiple x empty
      // Large object rings: empty x multiple
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<LC, LC>(alloc);
        auto* r2 = alloc_region<XC, XC>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Arena linked lists: multiple x singleton
      // Large object rings: empty x singleton
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<LC, LC>(alloc);
        auto* r2 = alloc_region<C, XC>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }

      // Arena linked lists: multiple x multiple
      // Large object rings: empty x empty
      {
        auto* alloc = ThreadAlloc::get();
        auto* r1 = alloc_region<LC, LC>(alloc);
        auto* r2 = alloc_region<LC, LC>(alloc);
        test_merge_helper<region_type>(alloc, r1, r2);
      }
    }
  }

  void run_test()
  {
    test_merge<RegionType::Trace>();
    test_merge<RegionType::Arena>();
  }
}