// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"
#include "region_arena.h"
#include "region_base.h"
#include "region_rc.h"
#include "region_trace.h"

namespace verona::rt
{
  inline thread_local RegionBase* active_region_md = nullptr;

  /**
   * Conceptually, objects are allocated within a region, and regions are
   * owned by cowns. Different regions may have different memory management
   * strategies, such as trace-based garbage collection, arena-based bump
   * allocation, or limited reference counting.
   *
   * The "entry point" or "root" of a region is an "Iso" (or isolated) object.
   * This Iso object is the only way to refer to a region, apart from external
   * references to objects inside a region.
   *
   * In the implementation, the Iso object points to a region metadata object.
   * This metadata object keeps track of the specific memory management
   * strategy and every object allocated within that region. It also contains
   * the remembered set and external reference table. The region metadata
   * object is always created, even if the only object in the region is the Iso
   * object.
   *
   * Because of circular dependencies, the implementation of regions is split
   * into multiple files:
   *
   *                            RegionBase (region_base.h)
   *                                       ^
   *                                       |
   *   Concrete region         RegionTrace (region_trace.h)
   *   implementations         RegionArena (region_arena.h)
   *                                       ^
   *                                       |
   *                                Region (region.h)
   *
   * RegionBase is the base class that contains all the common functionality
   * for regions, i.e. the remembered set and external reference table. All
   * region implementations inherit from RegionBase, and pointers to RegionBase
   * are passed around when the specific region type is unknown. As the base
   * class, RegionBase cannot refer to anything in the other region classes.
   *
   * The concrete region implementations are what actually implement the
   * memory management strategy for a region. They all inherit from RegionBase
   * and need to know the complete type of RegionBase. Anything that wants to
   * use a region will need to cast to the correct region implementation class.
   *
   * This class, Region, is just a collection of static helper methods. Thus,
   * it needs to know the complete types of RegionBase and the region
   * implementations, and none of the other region classes can refer to this
   * one.
   **/

  /**
   * NOTE: Search for "TODO(region)" for outstanding todos.
   **/
  using namespace snmalloc;

  /**
   * Helpers to convert a RegionType enum to a class at compile time.
   *
   * Example usage:
   *   using RegionClass = typename RegionType_to_class<region_type>::T;
   **/
  template<RegionType region_type>
  struct RegionType_to_class
  {
    using T = void;
  };

  template<>
  struct RegionType_to_class<RegionType::Trace>
  {
    using T = RegionTrace;
  };

  template<>
  struct RegionType_to_class<RegionType::Arena>
  {
    using T = RegionArena;
  };

  template<>
  struct RegionType_to_class<RegionType::Rc>
  {
    using T = RegionRc;
  };

  class Region
  {
  public:
    // Don't accidentally instantiate a Region.
    Region() = delete;

    /**
     * Returns the type of region represented by the Iso object `o`.
     **/
    static RegionType get_type(Object* o)
    {
      if (RegionTrace::is_trace_region(o))
        return RegionType::Trace;
      else if (RegionArena::is_arena_region(o))
        return RegionType::Arena;
      else if (RegionRc::is_rc_region(o))
        return RegionType::Rc;

      abort();
    }

    /**
     * Scan the region to find all cowns, following pointers to immutables and
     * subregions. This is used to keep reachable cowns alive and prevent them
     * from being collected by the leak detector.
     **/
    static void cown_scan(Alloc& alloc, Object* o, EpochMark epoch)
    {
      ObjectStack f(alloc);
      ObjectStack recurse(alloc);
      recurse.push(o);
      while (!recurse.empty())
      {
        o = recurse.pop();
        assert(o->debug_is_iso());
        Systematic::cout() << "Region Scan: scanning region: " << o
                           << Systematic::endl;
        switch (Region::get_type(o->get_region()))
        {
          case RegionType::Trace:
            Region::cown_scan_internal<RegionTrace>(
              alloc, o, f, recurse, epoch);
            break;
          case RegionType::Arena:
            Region::cown_scan_internal<RegionArena>(
              alloc, o, f, recurse, epoch);
            break;
          default:
            abort();
        }
      }
    }

    /**
     * Release and deallocate the region represented by Iso object `o`.
     *
     * As we discover Iso pointers to other regions, we add them to our
     * worklist.
     **/
    static void release(Alloc& alloc, Object* o)
    {
      assert(o->debug_is_iso() || o->is_opened());
      ObjectStack collect(alloc);
      Region::release_internal(alloc, o, collect);

      while (!collect.empty())
      {
        o = collect.pop();
        assert(o->debug_is_iso());
        Region::release_internal(alloc, o, collect);
      }
    }

    /**
     * Returns the region metadata object for the given Iso object `o`.
     *
     * This method returns a RegionBase that will need to be cast to a
     * specific region implementation.
     **/
    static RegionBase* get(Object* o)
    {
      assert(o->debug_is_iso());
      return o->get_region();
    }

  private:
    /**
     * Trace the region's object graph, following external pointers to cowns,
     * immutables, and subregions. Note that this will find all cowns reachable
     * from *within* the region, which may include cowns not reachable from the
     * object graph.
     *
     * Discovered subregions are added to the `recurse` worklist.
     *
     * We use the iterator to visit every object, and then trace to follow
     * pointers to cowns, and immutables and subregions (which might contain
     * pointers to cowns).
     *
     **/
    template<class RegionType>
    static void cown_scan_internal(
      Alloc& alloc,
      Object* o,
      ObjectStack& f,
      ObjectStack& recurse,
      EpochMark epoch)
    {
      // First, iterate over all objects and trace them.
      auto reg = RegionType::get(o);
      for (auto b : *reg)
      {
        b->trace(f);
      }

      // Now process the pointers we traced.
      while (!f.empty())
      {
        Object* p = f.pop();
        switch (p->get_class())
        {
          case Object::UNMARKED:
          case Object::MARKED:
            break;

          case Object::SCC_PTR:
            p = p->immutable();
            [[fallthrough]];
          case Object::RC:
            Systematic::cout()
              << "Region Scan: reaches immutable: " << p << Systematic::endl;
            Immutable::mark_and_scan(alloc, p, epoch);
            break;

          case Object::ISO:
            if (o != p)
            {
              Systematic::cout()
                << "Region Scan: pushing subregion to worklist: " << p
                << Systematic::endl;
              recurse.push(p);
            }
            break;

          case Object::COWN:
            Systematic::cout()
              << "Region Scan: reaches cown: " << p << Systematic::endl;
            cown::mark_for_scan(p, epoch);
            break;

          default:
            assert(0);
        }
      }
    }

    /**
     * Internal method for releasing and deallocating regions, that takes
     * a worklist (represented by `f` and `collect`).
     *
     * We dispatch based on the type of region represented by `o`.
     **/
    static void release_internal(Alloc& alloc, Object* o, ObjectStack& collect)
    {
      auto r = o->get_region();
      switch (Region::get_type(r))
      {
        case RegionType::Trace:
          ((RegionTrace*)r)->release_internal(alloc, o, collect);
          return;
        case RegionType::Arena:
          ((RegionArena*)r)->release_internal(alloc, o, collect);
          return;
        case RegionType::Rc:
        {
          ((RegionRc*)r)->release_internal(alloc, o, collect);
          return;
        }
        default:
          abort();
      }
    }
  };

  inline size_t debug_get_ref_count(Object* o)
  {
    return o->get_ref_count();
  }
} // namespace verona::rt
