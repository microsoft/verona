// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_rc
{
  constexpr auto region_type = RegionType::Rc;
  using C = C1<region_type>;

  void test_region_vector()
  {
    auto& alloc = ThreadAlloc::get();
    auto* o = new (alloc) C;

    const int NUM_OBJECTS = 128;

    // Allocate a bunch of objects so that the region vector uses many
    // allocation blocks to track each object in the region.
    for (int i = 0; i < NUM_OBJECTS; i++)
    {
      auto* o1 = new (alloc, o) C;
      assert(RegionRc::debug_get_ref_count(o1) == 1);
    }

    // Iterate through the region vector and make sure that all the
    // refcounts make sense.
    auto reg = (RegionRc*)Region::get(o);
    RegionVector<ObjectCount, Alloc>::iterator iter(reg->get_trivial_vec());

    ObjectCount* rm1;
    ObjectCount* rm2;
    ObjectCount* rm3;

    int count = 0;
    for (auto object_count : iter)
    {
      assert(object_count->count == 1);

      if (count == 5)
        rm1 = object_count;

      if (count == 40)
        rm2 = object_count;

      if (count == 100)
        rm3 = object_count;
      count++;
    }

    assert(count == NUM_OBJECTS);

    reg->get_trivial_vec()->remove(rm3);
    reg->get_trivial_vec()->remove(rm2);
    reg->get_trivial_vec()->remove(rm1);

    // Check that the iterator skips over holes left in the region vector
    count = 0;
    for (auto object_count : iter)
    {
      UNUSED(object_count);
      count++;
    }
    assert(count == NUM_OBJECTS - 3);

    // Check that the freelist is used to fill existing holes before bump
    // allocation.
    auto* o2 = new (alloc, o) C;
    auto* o2_idx = RegionRc::debug_get_rv_index(o2);

    assert(RegionRc::debug_get_ref_count(o2) == 1);
    assert((uintptr_t)rm1 == (uintptr_t)o2_idx);

    auto* o3 = new (alloc, o) C;
    auto* o3_idx = RegionRc::debug_get_rv_index(o3);

    assert(RegionRc::debug_get_ref_count(o3) == 1);
    assert((uintptr_t)rm2 == (uintptr_t)o3_idx);

    auto* o4 = new (alloc, o) C;
    auto* o4_idx = RegionRc::debug_get_rv_index(o4);

    assert(RegionRc::debug_get_ref_count(o4) == 1);
    assert((uintptr_t)rm3 == (uintptr_t)o4_idx);

    reg->get_trivial_vec()->dealloc(alloc);

    count = 0;
    for (auto object_count : iter)
    {
      UNUSED(object_count);
      count++;
    }
    assert(count == 0);
  }

  void run_test()
  {
    test_region_vector();
  }
}
