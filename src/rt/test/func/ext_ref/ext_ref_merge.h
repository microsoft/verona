// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

namespace ext_ref_merge
{
  struct C : public V<C>
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
    using T = C;

    auto& alloc = ThreadAlloc::get();
    (void)alloc;

    auto r1 = new (region_type) T;
    ExternalRef* wref1;
    {
      UsingRegion ur(r1);
      r1->f1 = new T;
      wref1 = create_external_reference(r1->f1);
    }

    auto r2 = new (region_type) T;
    ExternalRef* wref2;
    {
      UsingRegion ur(r2);
      r2->f1 = new T;
      wref2 = create_external_reference(r2->f1);
    }

    {
      UsingRegion ur(r1);
      merge(r2);
      r1->f1->f1 = r2;
    }

    check(!r2->debug_is_iso());

    Immutable::release(alloc, wref1);
    Immutable::release(alloc, wref2);

    region_release(r1);
    // Don't release r2, it was deallocated during the merge.

    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  void run_test()
  {
    basic_test<RegionType::Trace>();
    basic_test<RegionType::Arena>();
  }
}
