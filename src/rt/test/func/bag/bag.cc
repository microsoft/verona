// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ds/bag.h"

#include "test/harness.h"
#include "test/log.h"
#include "unordered_set"
#include "verona.h"

using namespace snmalloc;
using namespace verona::rt;

void test_bag_base()
{
  using E = BagElem<uintptr_t, uintptr_t>;
  using B = BagBase<E, Alloc>;
  {
    auto& alloc = ThreadAlloc::get();

    B bag;
    for (auto entry : bag)
    {
      UNUSED(entry);
      check(false);
    }

    auto item = bag.insert({nullptr, 123}, alloc);

    // Check that iter is setup correctly when the index points to a hole.
    auto item2 = bag.insert({nullptr, 456}, alloc);
    bag.remove(item2);

    for (auto entry : bag)
    {
      check(entry == item);
    }
    bag.dealloc(alloc);
  }
  {
    auto& alloc = ThreadAlloc::get();

    B bag;

    const int NUM_OBJECTS = 127;

    // Allocate a bunch of objects so that the region vector uses many
    // allocation blocks to track each object in the region.
    for (uintptr_t i = 0; i < NUM_OBJECTS; i++)
    {
      bag.insert({nullptr, i}, alloc);
    }

    // Iterate through the bag and make sure that all the
    // entries make sense.

    E* rm1 = nullptr;
    E* rm2 = nullptr;
    E* rm3 = nullptr;

    uintptr_t count = 0;
    for (auto entry : bag)
    {
      check(entry->metadata == (NUM_OBJECTS - count - 1));

      if (count == 5)
        rm1 = entry;

      if (count == 40)
        rm2 = entry;

      if (count == 100)
        rm3 = entry;
      count++;
    }

    check(count == NUM_OBJECTS);

    std::unordered_set<uintptr_t> removed = {
      (uintptr_t)rm1, (uintptr_t)rm2, (uintptr_t)rm3};

    bag.remove(rm1);
    bag.remove(rm2);
    bag.remove(rm3);

    // Check that the iterator skips over holes left in the bag.
    count = 0;
    for (auto item : bag)
    {
      UNUSED(item);
      count++;
    }
    assert(count == NUM_OBJECTS - 3);

    // Check that the freelist is used to fill existing holes before bump
    // allocation.
    auto idx1 = bag.insert({nullptr, 123}, alloc);
    check(removed.count((uintptr_t)idx1));

    auto idx2 = bag.insert({nullptr, 456}, alloc);
    check(removed.count((uintptr_t)idx2));

    auto idx3 = bag.insert({nullptr, 789}, alloc);
    check(removed.count((uintptr_t)idx3));

    // And that new allocations go back to bump allocation
    auto idx4 = bag.insert({nullptr, 10}, alloc);
    check(!removed.count((uintptr_t)idx4));

    bag.dealloc(alloc);
  }
}

int main(int, char**)
{
  test_bag_base();
  return 0;
}
