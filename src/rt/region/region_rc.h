// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"
#include "region_arena.h"
#include "region_base.h"
#include "region_trace.h"

namespace verona::rt
{
  using namespace snmalloc;

  /**
   * Please see region.h for the full documentation.
   *
   * This is a concrete implementation of a region, specifically one with
   * reference counting. This class inherits from RegionBase, but it cannot call
   * any of the static methods in Region.
   *
   * In a rc region, all objects are tracked using a bag; where each element
   * contains a pointer to the object and its current reference count. The bag
   * is updated each time an object is allocated or deallacted.
   *
   * The RegionRc uses a spare bit in each object pointer in the bag
   * (`FINALIZER_MASK`) to quickly deduce whether an object is trivial or
   * non-trivial.
   *
   **/
  class RegionRc : public RegionBase
  {
    friend class Freeze;
    friend class Region;
    friend class RegionTrace;

  private:
    static constexpr uintptr_t FINALISER_MASK = 1 << 1;

    size_t entry_point_count = 1;

    // Objects which may be in a cycle, so will be checked by gc_cycles.
    // FIXME: Use two stacks to simulate per-block queue based behaviour.
    StackThin<Object, Alloc> lins_stack;

    // Memory usage in the region.
    size_t current_memory_used = 0;

    size_t region_size = 0;

    RegionRc() : RegionBase() {}

    static const Descriptor* desc()
    {
      static constexpr Descriptor desc = {
        vsizeof<RegionRc>, nullptr, nullptr, nullptr};

      return &desc;
    }

  public:
    inline static RegionRc* get(Object* o)
    {
      assert(o->debug_is_iso());
      assert(is_rc_region(o->get_region()));
      return (RegionRc*)o->get_region();
    }

    size_t get_region_size() const
    {
      return region_size;
    }

    inline static bool is_rc_region(Object* o)
    {
      return o->is_type(desc());
    }

    /**
     * Creates a new rc region by allocating Object `o` of type `desc`. The
     * object is initialised as the Iso object for that region, and points to a
     * newly created Region metadata object. Returns a pointer to `o`.
     *
     * The default template parameter `size = 0` is to avoid writing two
     * definitions which differ only in one line. This overload works because
     * every object must contain a descriptor, so 0 is not a valid size.
     **/
    template<size_t size = 0>
    static Object* create(Alloc& alloc, const Descriptor* desc)
    {
      void* p = alloc.alloc<vsizeof<RegionRc>>();
      Object* o = Object::register_object(p, RegionRc::desc());
      auto reg = new (o) RegionRc();
      reg->use_memory(desc->size);

      if constexpr (size == 0)
        p = alloc.alloc(desc->size);
      else
        p = alloc.alloc<size>();
      o = Object::register_object(p, desc);

      reg->region_size += 1;

      reg->init_next(o);
      o->init_iso();
      o->set_region(reg);

      assert(Object::debug_is_aligned(o));
      return o;
    }

    /**
     * Allocates an object `o` of type `desc` in the region represented by the
     * Iso object `in`. Returns a pointer to `o`.
     *
     * The default template parameter `size = 0` is to avoid writing two
     * definitions which differ only in one line. This overload works because
     * every object must contain a descriptor, so 0 is not a valid size.
     **/
    template<size_t size = 0>
    static Object* alloc(Alloc& alloc, RegionRc* reg, const Descriptor* desc)
    {
      assert((size == 0) || (size == desc->size));
      assert(reg != nullptr);

      void* p = nullptr;
      if constexpr (size == 0)
        p = alloc.alloc(desc->size);
      else
        p = alloc.alloc<size>();

      auto o = (Object*)Object::register_object(p, desc);
      assert(Object::debug_is_aligned(o));

      o->init_ref_count();

      // GC heuristics.
      reg->use_memory(desc->size);
      reg->region_size += 1;
      return o;
    }

    void open(Object* o)
    {
      assert(o->get_class() == RegionMD::ISO);
      o->init_iso_ref_count(entry_point_count);
    }

    void close(Object* o)
    {
      assert(o->get_class() == RegionMD::OPEN_ISO);
      entry_point_count = o->get_ref_count();
      o->set_region(this);
    }

    /// Increments the reference count of `o`. The object `in` is the entry
    /// point to the region that contains `o`.
    static void incref(Object* o)
    {
      o->incref_rc_region();

      // TODO: Currently we don't remove items from the Lins stack because
      // there is no way to find the object in the stack without an O(n) pass.
      // This is still technically correct, but inefficient.
    }

    /// Decrements the reference count of `o`. The object `in` is the entry
    /// point to the region that contains `o`. If `decref` is called on an
    /// object with only one reference, then the object will be deallocated.
    static bool decref(Alloc& alloc, Object* o, RegionRc* reg)
    {
      if (decref_inner(o))
      {
        dealloc_object(alloc, o, reg);
        return true;
      }

      if (o->get_rc_colour() != RcColour::BLACK)
      {
        o->set_rc_colour(RcColour::BLACK);
        reg->lins_stack.push(o, alloc);
      }
      return false;
    }

    /// Get the reference count of `o`. The object `in` is the entry point to
    /// the region that contains `o`.
    static size_t get_ref_count(Object* o)
    {
      return o->get_ref_count();
    }

    /** Removes cyclic garbage from the region.
     *
     * This uses the Lins cyclic reference counting algorithm: a local mark-scan
     * of the subgraph of objects which are potentially cyclic. This cycle
     * detection is based on two key observations: first, that cycles are formed
     * only by interior object field references; and, second, if decrefing all
     * such interior references causes a zero ref count on those objects, then
     * it is a cycle.
     *
     * The algorithm uses a stack of object pointers -- referred to as a "Lins
     * stack" -- as starting points to check for, (and remove), potential
     * cycles. The Lins stack is maintained by the region during incref/decref
     * calls: an object is pushed when decrefing it yields a non-zero count;
     * conversely, an object is removed from the Lins stack when it is increfed.
     *
     * The Lins algorithm is a three-phase sweep over the subgraph of an object
     * with potential cyclic reachability. For each object o in the Lins stack,
     * it does the following:
     *
     *  1. o's subgraph is traced and each object is marked red and decrefed.
     *  Marking an object red indicates that it is potentially garbage. Objects
     *  that are marked red but still have non-zero ref count are added to a
     *  jump stack to be checked during phase 2. This ensures that they are not
     *  prematurely reclaimed.
     *
     *  2. For each element in the jump stack with an RC>1, it's red subgraph is
     *  re-traced and each edge is increfed, with their colours restored to
     *  green.
     *
     *  3. o's subgraph is re-traced a final time, and any remaining red objects
     *  are deallocated.
     **/
    static void gc_cycles(Alloc& alloc, Object* o, RegionRc* reg)
    {
      assert(o->get_class() == RegionMD::OPEN_ISO);
      UNUSED(o);
      ObjectStack jump_stack(alloc);
      while (!reg->lins_stack.empty())
      {
        auto p = reg->lins_stack.pop(alloc);

        if (p->get_rc_colour() == RcColour::BLACK)
        {
          mark_red(alloc, p, reg, jump_stack);
          scan(alloc, p, reg, jump_stack);
        }
      }
    }

    /**
     * Release and deallocate all objects within the region represented by the
     * Iso Object `o`.
     *
     * Note: this does not release subregions. Use Region::release instead.
     **/
    void release_internal(Alloc& alloc, Object* o, ObjectStack& collect)
    {
      open(o);
      assert(o->get_class() == RegionMD::OPEN_ISO);
      if (!decref_inner(o))
      {
        abort();
      }

      ObjectStack dfs(alloc);
      o->trace(dfs);
      LinkedObjectStack gc;

      if (!o->is_trivial())
      {
        o->finalise(o, collect);
      }

      while (!dfs.empty())
      {
        Object* f = dfs.pop();
        switch (f->get_class())
        {
          case Object::OPEN_ISO:
            assert(0);
          case Object::MARKED:
          case Object::UNMARKED:
            // FIXME: we need to ensure decrefing removes from the lins q
            if (decref_inner(f))
            {
              f->trace(dfs);
              if (!f->is_trivial())
              {
                f->finalise(o, collect);
              }
              gc.push(f);
            }
            break;
          case Object::SCC_PTR:
            f->immutable();
            f->decref();
            break;
          case Object::RC:
            f->decref();
            break;
          case Object::COWN:
            f->decref_cown();
            break;
          case Object::ISO:
            assert(f != o);
            break;
          default:
            assert(0);
        }
      }

      // Clean up any cyclic garbage not reachable from the entry point.
      release_cycles(alloc, o, gc, collect);

      while (!gc.empty())
      {
        Object* o = gc.pop();
        o->destructor();
        o->dealloc(alloc);
      }

      // finally, close the region and destroy the ISO object.
      close(o);
      o->destructor();
      o->dealloc(alloc);
      dealloc(alloc);
    }

  private:
    void release_cycles(
      Alloc& alloc, Object* o, LinkedObjectStack& gc, ObjectStack& collect)
    {
      ObjectStack dfs(alloc);
      while (!lins_stack.empty())
      {
        dfs.push(lins_stack.pop(alloc));
      }
      while (!dfs.empty())
      {
        Object* p = dfs.pop();
        switch (p->get_class())
        {
          case Object::OPEN_ISO:
          case Object::MARKED:
            break;
          case Object::UNMARKED:
            if (!p->is_trivial())
            {
              p->finalise(o, collect);
            }
            p->trace(dfs);
            gc.push(p);
            p->mark();
            break;
          case Object::SCC_PTR:
            p->immutable();
            p->decref();
            break;
          case Object::RC:
            p->decref();
            break;
          case Object::COWN:
            p->decref_cown();
            break;
          default:
            assert(0);
        }
      }
    }

    /**
     * Mark an object as red (i.e. dead). This is recursive over o's fields
     * where the field points to an objects which is also an rc candidate.
     *
     * mark_red traces the entire subgraph of o, decrefing on each edge and
     * marking each subobject as red.
     *
     * Note that objects are marked red indiscriminately. In other words, an
     * object `o` will be marked red even if it has a reference count greater
     * than 0 (indicating a potential external reference is keeping it alive).
     *
     * This eagerness is based on the observation by Lins that objects in a
     * cycle can have a non-zero count because they are referenced by other
     * objects in the same subgraph which are yet to be processed. This means it
     * is usually a good idea to eagerly mark them red but add them to a "jump
     * stack" for their liveness to be confirmed later on. This is a performance
     * optimisation that can result in fewer passes over the graph.
     **/
    static void
    mark_red(Alloc& alloc, Object* o, Object* in, ObjectStack& jump_stack)
    {
      if (!o->is_rc_candidate())
      {
        return;
      }

      if (o->get_rc_colour() != RcColour::RED)
      {
        o->set_rc_colour(RcColour::RED);
      }

      ObjectStack dfs(alloc);
      o->trace(dfs);

      while (!dfs.empty())
      {
        Object* f = dfs.pop();
        if (!o->is_rc_candidate())
        {
          continue;
        }

        if (f->get_class() == RegionMD::ISO)
        {
          if (f != in)
          {
            // If f is the entry point to a subregion we don't want to trace
            // into it: the colour of `f` is enough to determine the fate of
            // the entire subregion.
            continue;
          }

          RegionRc* reg = get(f);
          reg->entry_point_count -= 1;
        }
        else
        {
          decref_inner(f);
        }

        if (f->get_rc_colour() == RcColour::RED)
        {
          // base case to stop us infinitely iterating a cycle.
          continue;
        }

        if (get_ref_count(f) > 0)
        {
          // f might still be live, so add it to the jump stack to confirm
          // later on.
          //
          // Checking if `f` already exists in the stack is expensive and
          // unnecessary: a JS entry is only interesting if it is red, after
          // which it will be restored to green, so duplicates will just be
          // skipped over.
          jump_stack.push(f);
        }

        f->set_rc_colour(RcColour::RED);
        f->trace(dfs);
      }
    }

    static void
    scan(Alloc& alloc, Object* o, RegionRc* reg, ObjectStack& jump_stack)
    {
      // If the RC is positive after all interior pointers in this subgraph
      // have been decrefed, then |o| is rooted by *at least one* live
      // reference. Hence, it must be made green and have its interior
      // refcounts restored.
      if (o->is_rc_candidate() && get_ref_count(o) > 0)
      {
        restore_green(alloc, o, reg);
        return;
      }

      while (!jump_stack.empty())
      {
        Object* p = jump_stack.pop();
        if (p->get_rc_colour() == RcColour::RED && get_ref_count(p) > 0)
        {
          restore_green(alloc, p, reg);
        }
      }

      dealloc_reds(alloc, o, reg);
    }

    /**
     * Mark the object o as green (i.e. live). This is recursive over o's fields
     * where the field is also an rc candidate.
     *
     * Green objects won't be collected, so this method is called on objects
     * known to have live, non-cyclic references after the mark_red phase.
     **/
    static void restore_green(Alloc& alloc, Object* o, Object* in)
    {
      o->set_rc_colour(RcColour::GREEN);
      ObjectStack dfs(alloc);
      o->trace(dfs);

      while (!dfs.empty())
      {
        Object* f = dfs.pop();

        if (f->get_class() == RegionMD::ISO && f != in)
        {
          // If f is the entry point to a subregion we don't want to trace
          // into it or incref.
          continue;
        }

        incref(f);
        if (f->is_rc_candidate() && f->get_rc_colour() != RcColour::GREEN)
        {
          f->set_rc_colour(RcColour::GREEN);
          f->trace(dfs);
        }
      }
    }

    /**
     * Deallocate o if it is red. This is recursive over o's fields
     * where the field is also red and an rc candidate.
     *
     * This is the final phase of the cylic reference count scan. This must only
     * be called after all potential live objects which were eagerly marked red
     * are restored to green via the jump stack.
     **/
    static void dealloc_reds(Alloc& alloc, Object* o, RegionRc* reg)
    {
      if (o->get_rc_colour() != RcColour::RED)
      {
        return;
      }

      ObjectStack dfs(alloc);
      LinkedObjectStack gc;
      ObjectStack sub_regions(alloc);
      o->trace(dfs);
      while (!dfs.empty())
      {
        Object* f = dfs.pop();
        switch (f->get_class())
        {
          case Object::OPEN_ISO:
            // There should always be an external reference to the region's ISO.
            assert(0);
            break;
          case Object::MARKED:
            break;
          case Object::UNMARKED:
            // We don't decref here because that would have already
            // happened during mark_red. We simply need to check if
            // this is red (i.e. garbage).
            if (f->get_rc_colour() == RcColour::RED)
            {
              // We have made the ISO not ISO, so don't need to pass it.
              f->finalise(nullptr, sub_regions);
              f->trace(dfs);
              gc.push(f);
              f->mark();
            }
            break;
          case Object::SCC_PTR:
            f->immutable();
            f->decref();
            break;
          case Object::RC:
            f->decref();
            break;
          case Object::COWN:
            f->decref_cown();
            break;
          case Object::ISO:
            // Deallocation should only happen on an opened region.
            sub_regions.push(f);
            break;
          default:
            assert(0);
        }
      }

      release_sub_regions(alloc, sub_regions);

      while (!gc.empty())
      {
        Object* o = gc.pop();
        reg->region_size -= 1;
        o->destructor();
        o->dealloc(alloc);
      }
    }

    inline static bool decref_inner(Object* o)
    {
      o->decref_rc_region();
      return (o->get_ref_count() == 0);
    }

    static void dealloc_object(Alloc& alloc, Object* o, RegionRc* reg)
    {
      // We first need to decref -- and potentially deallocate -- any object
      // pointed to through `o`'s fields.
      ObjectStack dfs(alloc);
      ObjectStack sub_regions(alloc);
      o->trace(dfs);
      LinkedObjectStack gc;

      gc.push(o);

      while (!dfs.empty())
      {
        Object* p = dfs.pop();
        switch (p->get_class())
        {
          case Object::OPEN_ISO:
            if (decref_inner(p))
            {
              // There should always be an external reference keeping the ISO
              // alive.
              abort();
            }
            break;
          case Object::MARKED:
          case Object::UNMARKED:
            if (decref_inner(p))
            {
              p->trace(dfs);
              // The ISO tag has been removed from the entry point.
              p->finalise(nullptr, sub_regions);
              gc.push(p);
            }
            break;
          case Object::SCC_PTR:
            p->immutable();
            p->decref();
            break;
          case Object::RC:
            p->decref();
            break;
          case Object::COWN:
            p->decref_cown();
            break;
          case Object::ISO:
            //            assert(p != in);
            sub_regions.push(p);
            break;
          default:
            assert(0);
        }
      }

      while (!gc.empty())
      {
        Object* o = gc.pop();
        reg->region_size -= 1;
        o->destructor();
        o->dealloc(alloc);
      }

      release_sub_regions(alloc, sub_regions);
    }

    static void release_sub_regions(Alloc& alloc, ObjectStack& sub_regions)
    {
      while (!sub_regions.empty())
      {
        Object* o = sub_regions.pop();
        assert(o->debug_is_iso());
        Logging::cout() << "Region RC: releasing unreachable subregion: " << o
                        << Logging::endl;

        RegionBase* r = o->get_region();
        // Unfortunately, we can't use Region::release_internal because of a
        // circular dependency between header files.
        if (RegionTrace::is_trace_region(r))
          ((RegionTrace*)r)->release_internal(alloc, o, sub_regions);
        else if (RegionArena::is_arena_region(r))
          ((RegionArena*)r)->release_internal(alloc, o, sub_regions);
        else if (RegionRc::is_rc_region(r))
          ((RegionRc*)r)->release_internal(alloc, o, sub_regions);
        else
          abort();
      }
    }

  private:
    void use_memory(size_t size)
    {
      current_memory_used += size;
    }
  };

} // namespace verona::rt
