// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "memory.h"

namespace memory_gc
{
  constexpr auto region_type = RegionType::Trace;
  using C = C1<region_type>;
  using F = F1<region_type>;
  using MC = MediumC2<region_type>;
  using MF = MediumF2<region_type>;
  using LC = LargeC2<region_type>;
  using LF = LargeF2<region_type>;
  using XC = XLargeC2<region_type>;
  using XF = XLargeF2<region_type>;

  using Cx = C3<region_type>;
  using Fx = F3<region_type>;

  template<class... T>
  void alloc_garbage_helper(Alloc& alloc, Object* o)
  {
    alloc_in_region<T...>(alloc, o);
    check(Region::debug_size(o) == 1 + sizeof...(T)); // o + T...
    RegionTrace::gc(alloc, o);
    check(Region::debug_size(o) == 1); // only o is left
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
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      alloc_garbage_helper<C, F, MC, MF, LC, LF, XC, XF>(alloc, o);

      alloc_garbage_helper<C, C, C, XF, XF, MC, LC, LF, F, F, XC>(alloc, o);

      alloc_garbage_helper<XC, XC, XC, XF, MC>(alloc, o);

      alloc_garbage_helper<F, F, F, C, C, C, C, F>(alloc, o);

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // Allocate a lot of objects that are all connected.
    // Then break the link and GC.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      auto* o1 = new (alloc, o) C;
      auto* o2 = new (alloc, o) C;
      auto* o3 = new (alloc, o) C;
      auto* o4 = new (alloc, o) C;
      auto* o5 = new (alloc, o) C;
      auto* o6 = new (alloc, o) C;
      auto* o7 = new (alloc, o) C;

      // Link them up in some weird order.
      o->f1 = o3;
      o3->f2 = o5;
      o5->f1 = o6;
      o6->f1 = o1;
      o1->f1 = o2;
      o2->f2 = o4;
      o4->f2 = o7;

      check(Region::debug_size(o) == 8);
      RegionTrace::gc(alloc, o);
      check(Region::debug_size(o) == 8); // nothing collected

      // Break a link but re-add it.
      o1->f1 = nullptr;
      o1->f2 = o2;
      RegionTrace::gc(alloc, o);
      check(Region::debug_size(o) == 8); // nothing collected

      // Now actually break a link.
      o5->f1 = nullptr;
      RegionTrace::gc(alloc, o);
      check(Region::debug_size(o) == 3);

      // Now break the first link.
      o->f1 = nullptr;
      RegionTrace::gc(alloc, o);
      check(Region::debug_size(o) == 1); // only o is left

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
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
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) C;

      auto* o1 = new (alloc, o) C;
      new (alloc, o) C;
      auto* o3 = new (alloc, o) C;
      new (alloc, o) C;
      new (alloc, o) C;
      auto* o6 = new (alloc, o) C;
      new (alloc, o) C;
      new (alloc, o) C;
      new (alloc, o) C;
      auto* o10 = new (alloc, o) C;
      new (alloc, o) C;
      new (alloc, o) C;
      new (alloc, o) C;
      new (alloc, o) C;
      auto* o15 = new (alloc, o) C;

      // skip none, skip 1, skip 2, skip 4, skip 5
      // o -> o1 -> o3 -> o6 -> o10 -> o15
      o->f1 = o1;
      o1->f1 = o3;
      o3->f1 = o6;
      o6->f1 = o10;
      o10->f1 = o15;

      // Now run a GC.
      check(Region::debug_size(o) == 16);
      RegionTrace::gc(alloc, o);
      check(Region::debug_size(o) == 6);

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // Primary ring with objects that need finalisers.
    // We'll put some gaps in the region's object ring.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) F;

      auto* o1 = new (alloc, o) F;
      new (alloc, o) F;
      auto* o3 = new (alloc, o) F;
      auto* o4 = new (alloc, o) F;
      new (alloc, o) F;
      auto* o6 = new (alloc, o) F;
      auto* o7 = new (alloc, o) F;
      auto* o8 = new (alloc, o) F;
      new (alloc, o) F;
      auto* o10 = new (alloc, o) F;
      auto* o11 = new (alloc, o) F;
      auto* o12 = new (alloc, o) F;
      auto* o13 = new (alloc, o) F;

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
      check(Region::debug_size(o) == 14);
      RegionTrace::gc(alloc, o);
      check(Region::debug_size(o) == 11);

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
    }

    // Two rings. First and last objects in secondary ring are garbage.
    // For good measure, we'll add some more garbage.
    {
      auto& alloc = ThreadAlloc::get();
      auto* o = new (alloc) Fx;

      new (alloc, o) Fx;
      new (alloc, o) Fx;
      auto* o3 = new (alloc, o) Fx;
      auto* o4 = new (alloc, o) Fx;
      new (alloc, o) Fx;
      auto* o6 = new (alloc, o) Fx;
      new (alloc, o) Fx;
      new (alloc, o) Fx;

      new (alloc, o) Cx;
      new (alloc, o) Cx;
      auto* o11 = new (alloc, o) Cx;
      auto* o12 = new (alloc, o) Cx;
      new (alloc, o) Cx;
      auto* o14 = new (alloc, o) Cx;
      new (alloc, o) Cx;
      new (alloc, o) Cx;

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
      check(Region::debug_size(o) == 17);
      RegionTrace::gc(alloc, o);
      check(Region::debug_size(o) == 7);

      Region::release(alloc, o);
      snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
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
    C* scc = new (alloc) C;
    scc->f1 = new (alloc, scc) C;
    scc->f1->f1 = scc;
    Freeze::apply(alloc, scc);

    // Check that it's valid.
    auto rr = scc->debug_immutable_root();
    UNUSED(rr);
    check(rr->debug_test_rc(1));
    check(scc->f1->debug_immutable_root() == rr);
    check(scc->f1->debug_test_rc(1));

    // Now create a region that has pointers to the frozen SCC.
    C* r = new (alloc) C;
    new (alloc, r) C; // some garbage
    r->f1 = scc;
    r->f2 = scc->f1;

    check(Region::debug_size(r) == 2);
    check(scc->debug_test_rc(1));
    RegionTrace::gc(alloc, r);
    check(Region::debug_size(r) == 1);
    check(scc->debug_test_rc(2)); // gc discovered reference to scc

    Immutable::release(alloc, scc);
    check(scc->debug_test_rc(1));
    Region::release(alloc, r);
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  /**
   * Different kinds of cycles, both reachable and unreachable from the root.
   **/
  void test_cycles()
  {
    auto& alloc = ThreadAlloc::get();
    auto* o = new (alloc) C;

    // Allocate some reachable objects.
    auto* o1 = new (alloc, o) C;
    auto* o2 = new (alloc, o) C;
    auto* o3 = new (alloc, o) C;
    auto* o4 = new (alloc, o) C;
    auto* o5 = new (alloc, o) C;

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
    check(Region::debug_size(o) == 6);
    RegionTrace::gc(alloc, o);
    check(Region::debug_size(o) == 6);

    // Allocate some garbage cycles.

    // self cycle
    auto* o6 = new (alloc, o) C;
    o6->f1 = o6;

    // o7 -> o8 -> o7
    auto* o7 = new (alloc, o) C;
    auto* o8 = new (alloc, o) C;
    o7->f1 = o8;
    o8->f1 = o7;

    // o9 -> o10 -> o10
    auto* o9 = new (alloc, o) C;
    auto* o10 = new (alloc, o) C;
    o9->f1 = o10;
    o10->f1 = o10;

    // o11 -> o12 -> o13 -> o12
    auto* o11 = new (alloc, o) C;
    auto* o12 = new (alloc, o) C;
    auto* o13 = new (alloc, o) C;
    o11->f1 = o12;
    o12->f1 = o13;
    o13->f1 = o12;

    // o14 -> o15 -> o16 -> o14
    auto* o14 = new (alloc, o) C;
    auto* o15 = new (alloc, o) C;
    auto* o16 = new (alloc, o) C;
    o14->f1 = o15;
    o15->f1 = o16;
    o16->f1 = o14;

    // Now run a GC.
    check(Region::debug_size(o) == 17);
    RegionTrace::gc(alloc, o);
    check(Region::debug_size(o) == 6);

    Region::release(alloc, o);
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  /**
   * Run a GC after merging two regions. We'll also throw in a swap root.
   **/
  void test_merge()
  {
    auto& alloc = ThreadAlloc::get();

    // Create the first region.
    auto* r1 = new (alloc) Cx;
    r1->c1 = new (alloc, r1) Cx;
    r1->c2 = new (alloc, r1) Cx;
    r1->c1->f1 = new (alloc, r1) Fx;
    alloc_in_region<Cx, Cx, Fx, Fx>(alloc, r1); // unreachable
    check(Region::debug_size(r1) == 8);

    // Create another region.
    auto* r2 = new (alloc) Fx;
    r2->f1 = new (alloc, r2) Fx;
    r2->f2 = new (alloc, r2) Fx;
    r2->f1->c1 = new (alloc, r2) Cx;
    alloc_in_region<Fx, Fx, Cx, Cx>(alloc, r2); // unreachable
    check(Region::debug_size(r2) == 8);

    // Link the two regions together. Make both reachable from the other.
    r1->f1 = r2;
    r2->c1 = r1;

    // Now merge them.
    RegionTrace::merge(alloc, r1, r2);
    check(!r2->debug_is_iso());

    // Run a GC.
    check(Region::debug_size(r1) == 16);
    RegionTrace::gc(alloc, r1);
    check(Region::debug_size(r1) == 8);

    // Alloc a few things, swap root, then alloc some more.
    alloc_in_region<Fx, Cx, Fx>(alloc, r1);
    RegionTrace::swap_root(r1, r2);
    check(!r1->debug_is_iso() && r2->debug_is_iso());
    alloc_in_region<Cx, Cx, Fx>(alloc, r2);

    // Run another GC.
    check(Region::debug_size(r2) == 14);
    RegionTrace::gc(alloc, r2);
    check(Region::debug_size(r2) == 8);

    Region::release(alloc, r2);
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  /**
   * Run GC after swap root operations.
   **/
  void test_swap_root()
  {
    auto& alloc = ThreadAlloc::get();

    auto* o = new (alloc) Cx;
    o->c1 = new (alloc, o) Cx;
    o->c2 = new (alloc, o) Cx;
    o->c1->f1 = new (alloc, o) Fx;
    o->f1 = new (alloc, o) Fx;
    auto* nroot = new (alloc, o) Fx;
    o->f1->f1 = nroot;
    nroot->c1 = new (alloc, o) Cx;
    nroot->f1 = new (alloc, o) Fx;
    nroot->f1->f1 = new (alloc, o) Fx;
    alloc_in_region<Cx, Fx, Cx, Fx>(alloc, o); // unreachable

    // Run a GC.
    check(Region::debug_size(o) == 13);
    RegionTrace::gc(alloc, o);
    check(Region::debug_size(o) == 9);

    // Swap root, but this creates garbage.
    RegionTrace::swap_root(o, nroot);
    check(!o->debug_is_iso() && nroot->debug_is_iso());

    // Run another GC.
    RegionTrace::gc(alloc, nroot);
    check(Region::debug_size(nroot) == 4);

    // Create another region.
    o = new (alloc) Cx;
    o->f1 = new (alloc, o) Fx;
    o->f1->c1 = new (alloc, o) Cx;
    auto* nnroot = new (alloc, o) Cx;
    o->f1->c2 = nnroot;
    check(Region::debug_size(o) == 4);

    // Merge the regions.
    RegionTrace::merge(alloc, nroot, o);
    nroot->c2 = o;
    check(Region::debug_size(nroot) == 8);

    // Swap root again.
    RegionTrace::swap_root(nroot, nnroot);
    check(!nroot->debug_is_iso() && nnroot->debug_is_iso());

    // Run another GC.
    check(Region::debug_size(nnroot) == 8);
    RegionTrace::gc(alloc, nnroot);
    check(Region::debug_size(nnroot) == 1);

    Region::release(alloc, nnroot);
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  void test_additional_roots()
  {
    Systematic::cout() << "Additional roots test" << std::endl;

    auto& alloc = ThreadAlloc::get();
    auto* o = new (alloc) C;
    Systematic::cout() << "Root" << o << std::endl;

    // Allocate some reachable objects.
    auto* o1 = new (alloc, o) C;
    Systematic::cout() << " other" << o1 << std::endl;
    auto* s1 = new (alloc, o) C;
    Systematic::cout() << " sub" << s1 << std::endl;
    o1->f1 = s1;
    auto* o2 = new (alloc, o) C;
    Systematic::cout() << " other" << o2 << std::endl;
    auto* s2 = new (alloc, o) C;
    Systematic::cout() << " sub" << s2 << std::endl;
    o2->f1 = s2;
    auto* o3 = new (alloc, o) C;
    Systematic::cout() << " other" << o3 << std::endl;
    auto* s3 = new (alloc, o) C;
    Systematic::cout() << " sub" << s2 << std::endl;
    o3->f1 = s3;
    auto* o4 = new (alloc, o) C;
    Systematic::cout() << " other" << o4 << std::endl;
    auto* s4 = new (alloc, o) C;
    Systematic::cout() << " sub" << s4 << std::endl;
    o4->f1 = s4;
    auto* o5 = new (alloc, o) C;
    Systematic::cout() << " other" << o5 << std::endl;
    auto* s5 = new (alloc, o) C;
    Systematic::cout() << " sub" << s5 << std::endl;
    o5->f1 = s5;

    RegionTrace::push_additional_root(o, o1, alloc);
    RegionTrace::push_additional_root(o, o2, alloc);
    RegionTrace::push_additional_root(o, o3, alloc);
    RegionTrace::push_additional_root(o, o4, alloc);
    RegionTrace::push_additional_root(o, o5, alloc);

    check(Region::debug_size(o) == 11);
    RegionTrace::gc(alloc, o);
    Systematic::cout() << Region::debug_size(o) << std::endl;
    check(Region::debug_size(o) == 11);

    RegionTrace::pop_additional_root(o, o5, alloc);
    RegionTrace::pop_additional_root(o, o4, alloc);

    // Run another GC.
    RegionTrace::gc(alloc, o);
    check(Region::debug_size(o) == 7);

    RegionTrace::pop_additional_root(o, o3, alloc);
    RegionTrace::pop_additional_root(o, o2, alloc);
    RegionTrace::pop_additional_root(o, o1, alloc);

    // Run another GC.
    RegionTrace::gc(alloc, o);
    check(Region::debug_size(o) == 1);
    Region::release(alloc, o);
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
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