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
  void test_merge_helper(Object* r1, Object* r2)
  {
    {
      UsingRegion r(r1);
      merge(r2);

      check(!r2->debug_is_iso());

      // Allocate a few more things to ensure our pointers are correct.
      new LargeC2;
      new LargeF2;
      new XLargeC2;
    }

    region_release(r1);
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    check(live_count == 0);
  }

  /**
   * Tests merging two regions.
   **/
  template<RegionType region_type>
  void test_merge()
  {
    using C = C1;
    using F = F1;
    using LC = LargeC2;
    using XC = XLargeC2;

    // Quick sanity check that merges two regions, allocates a few more
    // objects, swaps roots a few times, and allocates more objects. We're
    // checking that region internal structures are still sensible.
    {
      auto r1 = alloc_region<C>(region_type);
      XC* o1 = alloc_in_region<1, F, XC>(r1);

      auto r2 = alloc_region<F>(region_type);
      auto o2 = alloc_in_region<1, F, C, LC>(r2);

      {
        UsingRegion rr(r1);
        merge(r2);
      }

      check(r1->debug_is_iso() && !r2->debug_is_iso());

      {
        UsingRegion rr(r1);
        allocs<0, F, F, LC, XC>();

        set_entry_point(o1);

        allocs<0, C, C>();

        set_entry_point(r2);

        allocs<0, LC, F>();

        set_entry_point(o2);

        check(o2->debug_is_iso());

        allocs<0, F, LC>();
      }

      region_release(o2);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
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
        auto r1 = alloc_region<C>(region_type);
        auto r2 = alloc_region<F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {singleton x singleton} (mismatch)
      // Secondary ring: {empty x nonempty}
      {
        auto* r1 = alloc_region<C>(region_type);
        auto* r2 = alloc_region<F, C, C>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {singleton x singleton} (mismatch)
      // Secondary ring: {nonempty x empty}
      {
        auto* r1 = alloc_region<C, F, F>(region_type);
        auto* r2 = alloc_region<F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {singleton x singleton} (mismatch)
      // Secondary ring: {nonempty x nonempty}
      {
        auto* r1 = alloc_region<C, F, F>(region_type);
        auto* r2 = alloc_region<F, C, C>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {singleton x multiple}
      // Secondary ring: {empty x empty}
      {
        auto* r1 = alloc_region<C>(region_type);
        auto* r2 = alloc_region<C, C, C>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {singleton x multiple}
      // Secondary ring: {empty x nonempty}
      {
        auto* r1 = alloc_region<C>(region_type);
        auto* r2 = alloc_region<C, C, F, F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {singleton x multiple}
      // Secondary ring: {nonempty x empty}
      {
        auto* r1 = alloc_region<F, C, C>(region_type);
        auto* r2 = alloc_region<F, F, F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {singleton x multiple}
      // Secondary ring: {nonempty x nonempty}
      {
        auto* r1 = alloc_region<C, F, F>(region_type);
        auto* r2 = alloc_region<C, C, F, F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {multiple x singleton}
      // Secondary ring: {empty x empty}
      {
        auto* r1 = alloc_region<C, C, C>(region_type);
        auto* r2 = alloc_region<C>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {multiple x singleton}
      // Secondary ring: {empty x nonempty}
      {
        auto* r1 = alloc_region<C, C, C>(region_type);
        auto* r2 = alloc_region<C, F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {multiple x singleton}
      // Secondary ring: {nonempty x empty}
      {
        auto* r1 = alloc_region<F, F, C>(region_type);
        auto* r2 = alloc_region<F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {multiple x singleton}
      // Secondary ring: {nonempty x nonempty}
      {
        auto* r1 = alloc_region<C, C, F>(region_type);
        auto* r2 = alloc_region<C, F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {multiple x multiple} (mismatch)
      // Secondary ring: {empty x empty}
      {
        auto* r1 = alloc_region<C, C, C>(region_type);
        auto* r2 = alloc_region<F, F, F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {multiple x multiple} (mismatch)
      // Secondary ring: {empty x nonempty}
      {
        auto* r1 = alloc_region<F, F, F>(region_type);
        auto* r2 = alloc_region<C, C, F>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {multiple x multiple} (mismatch)
      // Secondary ring: {nonempty x empty}
      {
        auto* r1 = alloc_region<F, F, F, C>(region_type);
        auto* r2 = alloc_region<C, C>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Primary ring:   {multiple x multiple} (mismatch)
      // Secondary ring: {nonempty x nonempty}
      {
        auto* r1 = alloc_region<C, C, C, F, F>(region_type);
        auto* r2 = alloc_region<F, F, F, C, C>(region_type);
        test_merge_helper<region_type>(r1, r2);
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
        auto* r1 = alloc_region<XC, XC>(region_type);
        auto* r2 = alloc_region<XC, XC>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Arena linked lists: empty x singleton
      // Large object rings: multiple x singleton
      {
        auto* r1 = alloc_region<XC, XC>(region_type);
        auto* r2 = alloc_region<XC, C>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Arena linked lists: empty x multiple
      // Large object rings: multiple x empty
      {
        auto* r1 = alloc_region<XC, XC>(region_type);
        auto* r2 = alloc_region<LC, LC>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Arena linked lists: singleton x empty
      // Large object rings: singleton x multiple
      {
        auto* r1 = alloc_region<C, XC>(region_type);
        auto* r2 = alloc_region<XC, XC>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Arena linked lists: singleton x singleton
      // Large object rings: singleton x singleton
      {
        auto* r1 = alloc_region<C, XC>(region_type);
        auto* r2 = alloc_region<C, XC>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Arena linked lists: singleton x multiple
      // Large object rings: singleton x empty
      {
        auto* r1 = alloc_region<C, XC>(region_type);
        auto* r2 = alloc_region<LC, LC>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Arena linked lists: multiple x empty
      // Large object rings: empty x multiple
      {
        auto* r1 = alloc_region<LC, LC>(region_type);
        auto* r2 = alloc_region<XC, XC>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Arena linked lists: multiple x singleton
      // Large object rings: empty x singleton
      {
        auto* r1 = alloc_region<LC, LC>(region_type);
        auto* r2 = alloc_region<C, XC>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }

      // Arena linked lists: multiple x multiple
      // Large object rings: empty x empty
      {
        auto* r1 = alloc_region<LC, LC>(region_type);
        auto* r2 = alloc_region<LC, LC>(region_type);
        test_merge_helper<region_type>(r1, r2);
      }
    }
  }

  void run_test()
  {
    test_merge<RegionType::Trace>();
    test_merge<RegionType::Arena>();
  }
}