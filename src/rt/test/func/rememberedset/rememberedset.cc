// Copyright (c) Contributers to Project Verona. All rights reserved.
// Licensed under the MIT License.

#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

template<RegionType region_type>
struct C1 : public V<C1<region_type>, region_type>
{
  C1<region_type>* f1 = nullptr;
  C1<region_type>* f2 = nullptr;

  void trace(ObjectStack* st) const
  {
    if (f1 != nullptr)
      st->push(f1);

    if (f2 != nullptr)
      st->push(f2);
  }

  // trace_possibly_iso would mean this object might need finalisation!
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

  auto* alloc = ThreadAlloc::get();

  // This will be our root object.
  C1<region_type>* o1 = new (alloc) C1<region_type>;

  // To insert o2 and o3 into o1's remembered set, we need to freeze them.
  // Right now, we can only freeze RegionTrace objects.
  auto* o2 = new (alloc) C1<RegionType::Trace>;
  Freeze::apply(alloc, o2);
  assert(o2->debug_is_rc());
  auto* o3 = new (alloc) C1<RegionType::Trace>;
  Freeze::apply(alloc, o3);
  assert(o3->debug_is_rc());

  // f1 and f2 need to be of type C1<RegionType::Trace>, and this is only
  // used for the GC part of the test.
  if constexpr (region_type == RegionType::Trace)
  {
    o1->f1 = o2;
    o1->f2 = o3;
  }

  // Move from the stack to o1.
  RegionClass::template insert<YesTransfer>(alloc, o1, o2);
  assert(o2->debug_rc() == 1 && o3->debug_rc() == 1);
  RegionClass::template insert<NoTransfer>(alloc, o1, o3);
  assert(o2->debug_rc() == 1 && o3->debug_rc() == 2);

  Immutable::release(alloc, o3);
  assert(o2->debug_rc() == 1 && o3->debug_rc() == 1);

  if constexpr (region_type == RegionType::Trace)
  {
    RegionTrace::gc(alloc, o1);
    assert(o2->debug_rc() == 1 && o3->debug_rc() == 1);

    Immutable::acquire(o2);
    assert(o2->debug_rc() == 2 && o3->debug_rc() == 1);
    o1->f1 = nullptr;
    RegionTrace::gc(alloc, o1);
    assert(o2->debug_rc() == 1 && o3->debug_rc() == 1);

    Immutable::release(alloc, o2);
    // o2 is now gone.
  }

  Region::release(alloc, o1);

  snmalloc::current_alloc_pool()->debug_check_empty();
}

template<RegionType region_type>
void merge_test()
{
  using RegionClass = typename RegionType_to_class<region_type>::T;
  using T = C1<region_type>;

  auto* alloc = ThreadAlloc::get();
  (void)alloc;

  // Create two regions.
  auto r1 = new (alloc) T;
  auto r2 = new (alloc) T;

  // Create some immutables that we can refcount, i.e. freeze them.
  // Right now, we can only freeze RegionTrace objects
  auto create_imm = [alloc]() {
    auto* o = new (alloc) C1<RegionType::Trace>;
    Freeze::apply(alloc, o);
    assert(o->debug_is_rc());
    return o;
  };
  auto* o1 = create_imm();
  auto* o2 = create_imm();
  auto* o3 = create_imm();

  // f1 and f2 need to be of type C1<RegionType::Trace>, and this is only
  // used for the GC part of the test.
  if constexpr (region_type == RegionType::Trace)
  {
    r1->f1 = o1;
    r1->f2 = o3;
    r2->f1 = o2;
    r2->f2 = o3;
  }

  RegionClass::template insert<YesTransfer>(alloc, r1, o1);
  assert(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 1);
  RegionClass::insert(alloc, r1, o3);
  assert(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 2);

  RegionClass::template insert<YesTransfer>(alloc, r2, o2);
  assert(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 2);
  RegionClass::insert(alloc, r2, o3);
  assert(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 3);

  RegionClass::merge(alloc, r1, r2);
  assert(o1->debug_rc() == 1 && o2->debug_rc() == 1 && o3->debug_rc() == 2);

  if constexpr (region_type == RegionType::Trace)
  {
    r1->f2 = nullptr;
    RegionTrace::gc(alloc, r1);
    // o2 is now gone
    assert(o1->debug_rc() == 1 && o3->debug_rc() == 1);
  }

  Immutable::release(alloc, o3);
  Region::release(alloc, r1);
  // Don't release r2, it was deallocated during the merge.

  snmalloc::current_alloc_pool()->debug_check_empty();
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