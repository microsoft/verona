// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

namespace ext_ref_merge
{
  template<RegionType region_type>
  struct C : public V<C<region_type>, region_type>
  {
    C* f1 = nullptr;
    void trace(ObjectStack& st) const
    {
      if (f1 != nullptr)
        st.push(f1);
    }
  };

  template<RegionType region_type>
  void basic_test()
  {
    using RegionClass = typename RegionType_to_class<region_type>::T;
    using T = C<region_type>;

    auto& alloc = ThreadAlloc::get();
    (void)alloc;

    auto r1 = new (alloc) T;
    r1->f1 = new (alloc, r1) T;
    auto reg1 = Region::get(r1);
    auto wref1 = ExternalRef::create(reg1, r1->f1);

    auto r2 = new (alloc) T;
    r2->f1 = new (alloc, r2) T;
    auto reg2 = Region::get(r2);
    auto wref2 = ExternalRef::create(reg2, r2->f1);

    RegionClass::merge(alloc, r1, r2);

    r1->f1->f1 = r2;

    check(!r2->debug_is_iso());

    Immutable::release(alloc, wref1);
    Immutable::release(alloc, wref2);

    Region::release(alloc, r1);
    // Don't release r2, it was deallocated during the merge.

    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  void run_test()
  {
    basic_test<RegionType::Trace>();
    basic_test<RegionType::Arena>();
  }
}
