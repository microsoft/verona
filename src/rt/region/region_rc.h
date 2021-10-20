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

    RefCounts counts{};

    RefCount* entry_point_count = nullptr;

    // Memory usage in the region.
    size_t current_memory_used = 0;

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

      reg->init_next(o);
      o->init_iso();
      o->set_region(reg);

      auto rc = reg->track_object(o, alloc);
      reg->entry_point_count = rc;

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
    static Object* alloc(Alloc& alloc, Object* in, const Descriptor* desc)
    {
      assert((size == 0) || (size == desc->size));
      RegionRc* reg = get(in);

      assert(reg != nullptr);

      void* p = nullptr;
      if constexpr (size == 0)
        p = alloc.alloc(desc->size);
      else
        p = alloc.alloc<size>();

      auto o = (Object*)Object::register_object(p, desc);
      assert(Object::debug_is_aligned(o));

      o->set_ref_count(reg->track_object(o, alloc));

      // GC heuristics.
      reg->use_memory(desc->size);
      return o;
    }

    /// Increments the reference count of `o`. The object `in` is the entry
    /// point to the region that contains `o`.
    static void incref(Object* o, Object* in)
    {
      // FIXME: An extra branch is needed here because the first field in the
      // RegionMD of an ISO holds a pointer to the region description, so we
      // can't quickly access the refcount as we would an ordinary object.
      if (o == in)
      {
        RegionRc* reg = get(in);
        reg->entry_point_count->metadata += 1;
        return;
      }
      RefCount* rc = o->get_ref_count();
      rc->metadata += 1;
    }

    /// Decrements the reference count of `o`. The object `in` is the entry
    /// point to the region that contains `o`. If `decref` is called on an
    /// object with only one reference, then the object will be deallocated.
    static bool decref(Alloc& alloc, Object* o, Object* in)
    {
      if (o == in)
      {
        RegionRc* reg = get(in);
        reg->entry_point_count->metadata -= 1;
        return false;
      }
      if (decref_inner(o))
      {
        dealloc_object(alloc, o, in);
        return true;
      }
      return false;
    }

    /// Get the reference count of `o`. The object `in` is the entry point to
    /// the region that contains `o`.
    static uintptr_t get_ref_count(Object* o, Object* in)
    {
      if (o == in)
      {
        RegionRc* reg = get(in);
        return reg->entry_point_count->metadata;
      }
      RefCount* rc = o->get_ref_count();
      return rc->metadata;
    }

  private:
    inline static bool decref_inner(Object* o)
    {
      RefCount* rc = o->get_ref_count();
      if (rc->metadata == 1)
      {
        return true;
      }
      rc->metadata -= 1;
      return false;
    }

    static void dealloc_object(Alloc& alloc, Object* o, Object* in)
    {
      // We first need to decref -- and potentially deallocate -- any object
      // pointed to through `o`'s fields.
      ObjectStack dfs(alloc);
      ObjectStack sub_regions(alloc);
      o->trace(dfs);
      ObjectStack fin_q(alloc);

      RegionRc* reg = get(in);

      fin_q.push(o);

      while (!dfs.empty())
      {
        Object* p = dfs.pop();
        switch (p->get_class())
        {
          case Object::ISO:
            if (p == in)
            {
              reg->entry_point_count->metadata -= 1;
            }
            else
            {
              sub_regions.push(p);
            }
            break;
          case Object::MARKED:
          case Object::UNMARKED:
            if (decref_inner(p))
            {
              fin_q.push(p);
              p->trace(dfs);
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
          default:
            assert(0);
        }
      }

      while (!fin_q.empty())
      {
        auto p = fin_q.pop();
        RefCount* rc = p->get_ref_count();
        reg->counts.remove(rc);
        if (!p->is_trivial())
        {
          p->finalise(in, sub_regions);
        }
        // Unlike traced regions, we can deallocate this immediately after
        // finalization.
        p->destructor();
        p->dealloc(alloc);
      }

      // Finally, we release any regions which were held by ISO pointers from
      // this object.
      while (!sub_regions.empty())
      {
        o = sub_regions.pop();
        assert(o->debug_is_iso());
        Systematic::cout() << "Region RC: releasing unreachable subregion: "
                           << o << Systematic::endl;

        // Note that we need to dispatch because `r` is a different region
        // metadata object.
        RegionBase* r = o->get_region();
        assert(r != in);

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

    /**
     * Release and deallocate all objects within the region represented by the
     * Iso Object `o`.
     *
     * Note: this does not release subregions. Use Region::release instead.
     **/
    void release_internal(Alloc& alloc, Object* o, ObjectStack& collect)
    {
      assert(o->debug_is_iso());

      // Finalize the non-trivial objects
      for (auto rc : counts)
      {
        auto p = (uintptr_t)rc->object;
        if ((p & FINALISER_MASK) == FINALISER_MASK)
        {
          auto untagged = (Object*)((uintptr_t)p & ~FINALISER_MASK);
          untagged->finalise(o, collect);
        }
      }

      // Note: it is safe to iterate the bag to deallocate objects since
      // the bag only stores a pointer to the objects.
      for (auto rc : counts)
      {
        auto untagged = (Object*)(((uintptr_t)rc->object) & ~FINALISER_MASK);
        untagged->destructor();
        untagged->dealloc(alloc);
      }

      counts.dealloc(alloc);
      dealloc(alloc);
    }

  public:
    template<IteratorType type = AllObjects>
    class iterator
    {
      friend class RegionRc;

      static_assert(
        type == Trivial || type == NonTrivial || type == AllObjects);

      iterator(RegionRc* r, RefCounts::iterator counts) : reg(r), counts(counts)
      {
        counts = r->counts.begin();
        ptr = (*counts)->object;
        next();
      }

      iterator(RegionRc* r, RefCounts::iterator counts, Object* p)
      : reg(r), counts(counts), ptr(p)
      {}

    private:
      RegionRc* reg;
      RefCounts::iterator counts;
      Object* ptr;

      void step()
      {
        ++counts;
        ptr = (*counts)->object;
      }

      void next()
      {
        if constexpr (type == AllObjects)
        {
          ptr = (*counts)->object;
          return;
        }
        else if constexpr (type == NonTrivial)
        {
          while ((counts != counts.end()) && !((uintptr_t)ptr & FINALISER_MASK))
          {
            step();
          }
          ptr = (*counts)->object;
          return;
        }
        else
        {
          while ((counts != counts.end()) && ((uintptr_t)ptr & FINALISER_MASK))
          {
            step();
          }
          ptr = (*counts)->object;
          return;
        }
      }

    public:
      iterator operator++()
      {
        step();
        next();
        return *this;
      }

      inline bool operator!=(const iterator& other) const
      {
        assert(reg == other.reg);
        return ptr != other.ptr;
      }

      inline Object* operator*() const
      {
        return ptr;
      }
    };

    template<IteratorType type = AllObjects>
    inline iterator<type> begin()
    {
      return {this, counts.begin()};
    }

    template<IteratorType type = AllObjects>
    inline iterator<type> end()
    {
      return {this, counts.end(), nullptr};
    }

  private:
    void use_memory(size_t size)
    {
      current_memory_used += size;
    }

    RefCount* track_object(Object* o, Alloc& alloc)
    {
      auto tagged = (uintptr_t)o;
      if (!(o->is_trivial()))
        tagged |= FINALISER_MASK;
      return counts.insert({(Object*)tagged, 1}, alloc);
    }
  };

} // namespace verona::rt
