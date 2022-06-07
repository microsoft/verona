// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;
using namespace verona::rt::api;

struct C1 : public V<C1>
{
  C1* f1 = nullptr;
  C1* f2 = nullptr;

  void trace(ObjectStack& st) const
  {
    if (f1 != nullptr)
      st.push(f1);

    if (f2 != nullptr)
      st.push(f2);
  }
};

/**
 * Tests inserting into a region's remembered set.
 *
 * TODO(region): Not clear if it makes sense to freeze a RegionArena or insert
 * into its remembered set. This test will probably need to be completely
 * rewritten for RegionArena, once we have a story for how insert should work.
 **/
template<RegionType region_type>
void basic_test()
{
  using RegionClass = typename RegionType_to_class<region_type>::T;

  auto& alloc = ThreadAlloc::get();

  // This will be our root object.
  auto o1 = new (region_type) C1;

  // To insert o2 and o3 into o1's remembered set, we need to freeze them.
  // Right now, we can only freeze RegionTrace objects.
  auto o2 = new (RegionType::Trace) C1;
  freeze(o2);
  check(o2->debug_is_rc());

  auto o3 = new (RegionType::Trace) C1;
  freeze(o3);
  check(o3->debug_is_rc());

  // f1 and f2 need to be of type C1 in <RegionType::Trace>, and this is only
  // used for the GC part of the test.
  if constexpr (region_type == RegionType::Trace)
  {
    o1->f1 = o2;
    o1->f2 = o3;
  }

  // Move from the stack to o1.
  RegionClass::template insert<YesTransfer>(alloc, o1, o2);
  check(o2->debug_rc() == 1 && o3->debug_rc() == 1);
  RegionClass::template insert<NoTransfer>(alloc, o1, o3);
  check(o2->debug_rc() == 1 && o3->debug_rc() == 2);

  Immutable::release(alloc, o3);
  check(o2->debug_rc() == 1 && o3->debug_rc() == 1);

  if constexpr (region_type == RegionType::Trace)
  {
    RegionTrace::gc(alloc, o1);
    check(o2->debug_rc() == 1 && o3->debug_rc() == 1);

    Immutable::acquire(o2);
    check(o2->debug_rc() == 2 && o3->debug_rc() == 1);
    o1->f1 = nullptr;
    RegionTrace::gc(alloc, o1);
    check(o2->debug_rc() == 1 && o3->debug_rc() == 1);

    Immutable::release(alloc, o2);
    // o2 is now gone.
  }

  region_release(o1);

  snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
}

template<RegionType region_type>
void merge_test()
{
  using RegionClass = typename RegionType_to_class<region_type>::T;

  auto& alloc = ThreadAlloc::get();

  // Create two regions.
  auto r1 = new (region_type) C1;
  auto r2 = new (region_type) C1;

  // Create some immutables that we can refcount, i.e. freeze them.
  // Right now, we can only freeze RegionTrace objects
  auto create_imm = []() {
    auto* o = new (RegionType::Trace) C1;
    freeze(o);
    check(o->debug_is_rc());
    return o;
  };
  auto* o1 = create_imm();
  auto* o2 = create_imm();
  auto* o3 = create_imm();

  // f1 and f2 need to be of type C1 in <RegionType::Trace>, and this is only
  // used for the GC part of the test.
  if constexpr (region_type == RegionType::Trace)
  {
    r1->f1 = o1;
    r1->f2 = o3;
    r2->f1 = o2;
    r2->f2 = o3;
  }

  RegionClass::template insert<YesTransfer>(alloc, r1, o1);
  check(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 1);
  RegionClass::insert(alloc, r1, o3);
  check(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 2);

  RegionClass::template insert<YesTransfer>(alloc, r2, o2);
  check(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 2);
  RegionClass::insert(alloc, r2, o3);
  check(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 3);

  {
    UsingRegion rr(r1);
    merge(r2);
  }
  check(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 2);

  if constexpr (region_type == RegionType::Trace)
  {
    r1->f2 = nullptr;
    RegionTrace::gc(alloc, r1);
    // o2 is now gone
    check(o1->debug_rc() == 1 && o3->debug_rc() == 1);
  }

  Immutable::release(alloc, o3);
  region_release(r1);
  // Don't release r2, it was deallocated during the merge.

  snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
}

int main(int argc, char** argv)
{
  (void)argc;
  (void)argv;

  basic_test<RegionType::Trace>();
  basic_test<RegionType::Arena>();

  merge_test<RegionType::Trace>();
  merge_test<RegionType::Arena>();

  return 0;
}