// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_gc
{
  constexpr auto region_type = RegionType::Trace;
  using C = C1;
  using F = F1;
  using MC = MediumC2;
  using MF = MediumF2;
  using LC = LargeC2;
  using LF = LargeF2;
  using XC = XLargeC2;
  using XF = XLargeF2;

  using Cx = C3;
  using Fx = F3;

  template<class... T>
  void alloc_garbage_helper()
  {
    allocs<0, T...>();
    check(debug_size() == 1 + sizeof...(T)); // o + T...
    region_collect();
    check(debug_size() == 1); // only o is left
  }

  /**
   * A few basic tests to start:
   *   - allocating only unreachable objects, which should all get collected.
   *   - allocating objects and making them all reachable, but then breaking
   *     the "links" and running GCs.
   **/
  void test_basic()
  {
    // Allocate a lot of garbage.
    {
      auto* o = new (RegionType::Trace) C;

      {
        UsingRegion rr(o);

        alloc_garbage_helper<C, F, MC, MF, LC, LF, XC, XF>();

        alloc_garbage_helper<C, C, C, XF, XF, MC, LC, LF, F, F, XC>();

        alloc_garbage_helper<XC, XC, XC, XF, MC>();

        alloc_garbage_helper<F, F, F, C, C, C, C, F>();
      }

      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
    }

    // Allocate a lot of objects that are all connected.
    // Then break the link and GC.
    {
      auto* o = new (RegionType::Trace) C;

      {
        UsingRegion rr(o);

        auto* o1 = new C;
        auto* o2 = new C;
        auto* o3 = new C;
        auto* o4 = new C;
        auto* o5 = new C;
        auto* o6 = new C;
        auto* o7 = new C;

        // Link them up in some weird order.
        o->f1 = o3;
        o3->f2 = o5;
        o5->f1 = o6;
        o6->f1 = o1;
        o1->f1 = o2;
        o2->f2 = o4;
        o4->f2 = o7;

        check(debug_size() == 8);
        region_collect();
        check(debug_size() == 8); // nothing collected

        // Break a link but re-add it.
        o1->f1 = nullptr;
        o1->f2 = o2;
        region_collect();
        check(debug_size() == 8); // nothing collected

        // Now actually break a link.
        o5->f1 = nullptr;
        region_collect();
        check(debug_size() == 3);

        // Now break the first link.
        o->f1 = nullptr;
        region_collect();
        check(debug_size() == 1); // only o is left
      }

      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
    }
  }

  /**
   * Allocating "linked lists" and then running GCs.
   *
   * The object graph has the shape of a linked list, where each object has a
   * pointer to at most one object.
   *
   * In particular, we're interested in the shape of the region's internal
   * structures.
   **/
  void test_linked_list()
  {
    // Primary ring with objects that don't need finalisers.
    // We'll put some gaps in the region's object ring.
    {
      auto* o = new (RegionType::Trace) C;

      {
        UsingRegion rr(o);

        auto* o1 = new C;
        new C;
        auto* o3 = new C;
        new C;
        new C;
        auto* o6 = new C;
        new C;
        new C;
        new C;
        auto* o10 = new C;
        new C;
        new C;
        new C;
        new C;
        auto* o15 = new C;

        // skip none, skip 1, skip 2, skip 4, skip 5
        // o -> o1 -> o3 -> o6 -> o10 -> o15
        o->f1 = o1;
        o1->f1 = o3;
        o3->f1 = o6;
        o6->f1 = o10;
        o10->f1 = o15;

        // Now run a GC.
        check(debug_size() == 16);
        region_collect();
        check(debug_size() == 6);
      }

      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
    }

    // Primary ring with objects that need finalisers.
    // We'll put some gaps in the region's object ring.
    {
      auto* o = new (RegionType::Trace) F;

      {
        UsingRegion rr(o);
        auto* o1 = new F;
        new F;
        auto* o3 = new F;
        auto* o4 = new F;
        new F;
        auto* o6 = new F;
        auto* o7 = new F;
        auto* o8 = new F;
        new F;
        auto* o10 = new F;
        auto* o11 = new F;
        auto* o12 = new F;
        auto* o13 = new F;

        // string of 1, skip, string of 2, skip, string of 3, etc.
        // o -> o1 -> o3 -> o4 -> o6 -> o7 -> o8 -> o10 -> o11 -> o12 -> o13
        o->f1 = o1;
        o1->f1 = o3;
        o3->f1 = o4;
        o4->f1 = o6;
        o6->f1 = o7;
        o7->f1 = o8;
        o8->f1 = o10;
        o10->f1 = o11;
        o11->f1 = o12;
        o12->f1 = o13;

        // Now run a GC.
        check(debug_size() == 14);
        region_collect();
        check(debug_size() == 11);
      }

      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
    }

    // Two rings. First and last objects in secondary ring are garbage.
    // For good measure, we'll add some more garbage.
    {
      auto* o = new (RegionType::Trace) Fx;

      {
        UsingRegion rr(o);
        new Fx;
        new Fx;
        auto* o3 = new Fx;
        auto* o4 = new Fx;
        new Fx;
        auto* o6 = new Fx;
        new Fx;
        new Fx;

        new Cx;
        new Cx;
        auto* o11 = new Cx;
        auto* o12 = new Cx;
        new Cx;
        auto* o14 = new Cx;
        new Cx;
        new Cx;

        // First and last allocated objects will be first and last in the rings.
        // And we'll also leave a gap in the middle.
        // o -> o3 -> o4 -> o6 -> o11 -> o12 -> o14
        o->f1 = o3;
        o3->f1 = o4;
        o4->f1 = o6;
        o6->c1 = o11;
        o11->c1 = o12;
        o12->c1 = o14;

        // Now run a GC.
        check(debug_size() == 17);
        region_collect();
        check(debug_size() == 7);
      }
      region_release(o);
      snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
    }
  }

  /**
   * Create an immutable SCC and reference it from a region. We reference both
   * the "root" of the SCC and also some other object in that graph.
   * A GC should find the SCC and incref it.
   **/
  void test_freeze()
  {
    auto& alloc = ThreadAlloc::get();

    // Create and freeze an SCC.
    C* scc = new (RegionType::Trace) C;
    {
      UsingRegion rr(scc);
      scc->f1 = new C;
      scc->f1->f1 = scc;
    }
    scc = freeze(scc);

    // Check that it's valid.
    auto rr = scc->debug_immutable_root();
    UNUSED(rr);
    check(rr->debug_test_rc(1));
    check(scc->f1->debug_immutable_root() == rr);
    check(scc->f1->debug_test_rc(1));

    // Now create a region that has pointers to the frozen SCC.
    C* r = new (RegionType::Trace) C;
    {
      UsingRegion rr(r);
      new C; // some garbage
      r->f1 = scc;
      r->f2 = scc->f1;

      check(debug_size() == 2);
      check(scc->debug_test_rc(1));
      region_collect();
      check(debug_size() == 1);
      check(scc->debug_test_rc(2)); // gc discovered reference to scc

      Immutable::release(alloc, scc);
      check(scc->debug_test_rc(1));
    }
    region_release(r);
    snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
  }

  /**
   * Different kinds of cycles, both reachable and unreachable from the root.
   **/
  void test_cycles()
  {
    auto* o = new (RegionType::Trace) C;

    {
      UsingRegion rr(o);
      // Allocate some reachable objects.
      auto* o1 = new C;
      auto* o2 = new C;
      auto* o3 = new C;
      auto* o4 = new C;
      auto* o5 = new C;

      // self cycles: o, o1
      o->f2 = o;
      o1->f2 = o1;

      // cycle: o -> o1 -> o2 -> o3 -> o4 -> o5 -> o
      o->f1 = o1;
      o1->f1 = o2;
      o2->f1 = o3;
      o3->f1 = o4;
      o4->f1 = o5;
      o5->f1 = o;

      // back edge causing cycle: o4 -> o2
      o4->f2 = o2;

      // Now run a GC.
      check(debug_size() == 6);
      region_collect();
      check(debug_size() == 6);

      // Allocate some garbage cycles.

      // self cycle
      auto* o6 = new C;
      o6->f1 = o6;

      // o7 -> o8 -> o7
      auto* o7 = new C;
      auto* o8 = new C;
      o7->f1 = o8;
      o8->f1 = o7;

      // o9 -> o10 -> o10
      auto* o9 = new C;
      auto* o10 = new C;
      o9->f1 = o10;
      o10->f1 = o10;

      // o11 -> o12 -> o13 -> o12
      auto* o11 = new C;
      auto* o12 = new C;
      auto* o13 = new C;
      o11->f1 = o12;
      o12->f1 = o13;
      o13->f1 = o12;

      // o14 -> o15 -> o16 -> o14
      auto* o14 = new C;
      auto* o15 = new C;
      auto* o16 = new C;
      o14->f1 = o15;
      o15->f1 = o16;
      o16->f1 = o14;

      // Now run a GC.
      check(debug_size() == 17);
      region_collect();
      check(debug_size() == 6);
    }

    region_release(o);
    snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
  }

  /**
   * Run a GC after merging two regions. We'll also throw in a swap root.
   **/
  void test_merge()
  {
    // Create the first region.
    auto* r1 = new (RegionType::Trace) Cx;
    {
      UsingRegion rr(r1);

      r1->c1 = new Cx;
      r1->c2 = new Cx;
      r1->c1->f1 = new Fx;
      allocs<0, Cx, Cx, Fx, Fx>(); // unreachable
      check(debug_size() == 8);
    }

    // Create another region.
    auto* r2 = new (RegionType::Trace) Fx;
    {
      UsingRegion rr(r2);
      r2->f1 = new Fx;
      r2->f2 = new Fx;
      r2->f1->c1 = new Cx;
      allocs<0, Fx, Fx, Cx, Cx>(); // unreachable
      check(debug_size() == 8);
    }

    {
      UsingRegion rr(r1);
      merge(r2);

      // Link the two regions together. Make both reachable from the other.
      r1->f1 = r2;
      r2->c1 = r1;

      check(!r2->debug_is_iso());

      // Run a GC.
      check(debug_size() == 16);
      region_collect();
      check(debug_size() == 8);

      // Alloc a few things, swap root, then alloc some more.
      allocs<0, Fx, Cx, Fx>();
      set_entry_point(r2);
      check(!r1->debug_is_iso() && r2->debug_is_iso());
      allocs<0, Cx, Cx, Fx>();

      // Run another GC.
      check(debug_size() == 14);
      region_collect();
      check(debug_size() == 8);
    }

    region_release(r2);
    snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
  }

  /**
   * Run GC after swap root operations.
   **/
  void test_swap_root()
  {
    Cx* nnroot;

    auto* o = new (RegionType::Trace) Cx;
    {
      UsingRegion rr(o);
      o->c1 = new Cx;
      o->c2 = new Cx;
      o->c1->f1 = new Fx;
      o->f1 = new Fx;
      auto* nroot = new Fx;
      o->f1->f1 = nroot;
      nroot->c1 = new Cx;
      nroot->f1 = new Fx;
      nroot->f1->f1 = new Fx;
      allocs<0, Cx, Fx, Cx, Fx>(); // unreachable

      // Run a GC.
      check(debug_size() == 13);
      region_collect();
      check(debug_size() == 9);

      // Swap root, but this creates garbage.
      set_entry_point(nroot);
      check(!o->debug_is_iso() && nroot->debug_is_iso());

      // Run another GC.
      region_collect();
      check(debug_size() == 4);

      // Create another region.
      o = new (RegionType::Trace) Cx;
      {
        UsingRegion rr2(o);
        o->f1 = new Fx;
        o->f1->c1 = new Cx;
        nnroot = new Cx;
        o->f1->c2 = nnroot;
        check(debug_size() == 4);
      }

      // Merge the regions.
      merge(o);
      nroot->c2 = o;
      check(debug_size() == 8);

      // Swap root again.
      set_entry_point(nnroot);
      check(!nroot->debug_is_iso() && nnroot->debug_is_iso());

      // Run another GC.
      check(debug_size() == 8);
      region_collect();
      check(debug_size() == 1);
    }

    region_release(nnroot);
    snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
  }

  void test_additional_roots()
  {
    Logging::cout() << "Additional roots test" << std::endl;

    auto& alloc = ThreadAlloc::get();
    auto* o = new (RegionType::Trace) C;
    Logging::cout() << "Root" << o << std::endl;
    {
      UsingRegion rr(o);
      // Allocate some reachable objects.
      auto* o1 = new C;
      Logging::cout() << " other" << o1 << std::endl;
      auto* s1 = new C;
      Logging::cout() << " sub" << s1 << std::endl;
      o1->f1 = s1;
      auto* o2 = new C;
      Logging::cout() << " other" << o2 << std::endl;
      auto* s2 = new C;
      Logging::cout() << " sub" << s2 << std::endl;
      o2->f1 = s2;
      auto* o3 = new C;
      Logging::cout() << " other" << o3 << std::endl;
      auto* s3 = new C;
      Logging::cout() << " sub" << s2 << std::endl;
      o3->f1 = s3;
      auto* o4 = new C;
      Logging::cout() << " other" << o4 << std::endl;
      auto* s4 = new C;
      Logging::cout() << " sub" << s4 << std::endl;
      o4->f1 = s4;
      auto* o5 = new C;
      Logging::cout() << " other" << o5 << std::endl;
      auto* s5 = new C;
      Logging::cout() << " sub" << s5 << std::endl;
      o5->f1 = s5;

      RegionTrace::push_additional_root(o, o1, alloc);
      RegionTrace::push_additional_root(o, o2, alloc);
      RegionTrace::push_additional_root(o, o3, alloc);
      RegionTrace::push_additional_root(o, o4, alloc);
      RegionTrace::push_additional_root(o, o5, alloc);

      check(debug_size() == 11);
      region_collect();
      Logging::cout() << debug_size() << std::endl;
      check(debug_size() == 11);

      RegionTrace::pop_additional_root(o, o5, alloc);
      RegionTrace::pop_additional_root(o, o4, alloc);

      // Run another GC.
      region_collect();
      check(debug_size() == 7);

      RegionTrace::pop_additional_root(o, o3, alloc);
      RegionTrace::pop_additional_root(o, o2, alloc);
      RegionTrace::pop_additional_root(o, o1, alloc);

      // Run another GC.
      region_collect();
      check(debug_size() == 1);
    }
    region_release(o);
    snmalloc::debug_check_empty<snmalloc::Alloc::Config>();
  }

  void run_test()
  {
    test_basic();
    test_additional_roots();
    test_linked_list();
    test_freeze();
    test_cycles();
    test_merge();
    test_swap_root();
  }
}