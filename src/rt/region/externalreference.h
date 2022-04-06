// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"
#include "ds/hashmap.h"
#include "immutable.h"

#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;

  class RememberedSet;

  class ExternalReferenceTable
  {
  public:
    /**
     * An external reference is a pointer to a ExternalRef object. There is at
     * most one ExternalRef object for each object in a region. An external
     * reference can be used to, in constant time, find a specific object in a
     * region.
     */
    class ExternalRef : public Object
    {
      friend class ExternalReferenceTable;

    private:
      // The `ExternalReferenceTable` of the region where `o` lives
      std::atomic<ExternalReferenceTable*> ert;
      // The object externally referred to
      Object* o;

      static void gc_trace(const Object*, ObjectStack&) {}

      static const Descriptor* desc()
      {
        static constexpr Descriptor desc = {
          vsizeof<ExternalRef>, gc_trace, nullptr, nullptr};

        return &desc;
      }

      // May only be called if there's a ext_ref for o in rs.
      static ExternalRef*
      find_ext_ref(ExternalReferenceTable* ert, const Object* o)
      {
        auto i = ert->external_map->find(o);
        assert(i != ert->external_map->end());
        assert(i.value());
        return i.value();
      }

      ExternalRef(ExternalReferenceTable* ert_, Object* o_)
      : Object(), ert{ert_}, o{o_}
      {
        make_scc();

        incref();
        ert.load(std::memory_order_relaxed)
          ->insert(ThreadAlloc::get(), o, this);

        o->set_has_ext_ref();
      }

    public:
      /**
       * Creating an external reference to `o` in `region`.
       */
      static ExternalRef* create(ExternalReferenceTable* ert, Object* o)
      {
        assert(!o->debug_is_immutable() && !o->debug_is_cown());
        if (o->has_ext_ref())
        {
          auto ext_ref = find_ext_ref(ert, o);
          ext_ref->incref();
          return ext_ref;
        }

        // External references are not allocated in any regions, but have
        // independent lifetime protected by reference counting.
        void* header_obj =
          ThreadAlloc::get().template alloc<vsizeof<ExternalRef>>();
        Object* obj = Object::register_object(header_obj, desc());
        return new (obj) ExternalRef(ert, o);
      }

      /**
       * May only be called when `is_in` returns `true`.
       */
      Object* get()
      {
        assert(o);
        return o;
      }

      /**
       * Check if this external reference still points to an object in `region`.
       */
      bool is_in(ExternalReferenceTable* ert_)
      {
        return ert.load(std::memory_order_relaxed) == ert_;
      }
    };

    // No tracing is need for external_map, because entries in the map doesn't
    // contribute to objects RC; when an object is collected, its corresponding
    // entry in the map (if any) is removed as well.
    using ExternalMap = ObjectMap<std::pair<Object*, ExternalRef*>>;

    ExternalMap* external_map;

  public:
    ExternalReferenceTable()
    : external_map(ExternalMap::create(ThreadAlloc::get()))
    {}

    void dealloc(Alloc& alloc)
    {
      for (auto it = external_map->begin(); it != external_map->end(); ++it)
        remove_ref(alloc, it);

      external_map->dealloc(alloc);
      alloc.dealloc<sizeof(ExternalMap)>(external_map);
    }

    void merge(Alloc& alloc, ExternalReferenceTable* that)
    {
      for (auto e : *that->external_map)
      {
        auto* ext_ref = *e.second;
        assert(ext_ref->o);
        ext_ref->ert.store(this, std::memory_order_relaxed);
        *e.second = nullptr;
        insert(alloc, e.first, ext_ref);
      }
    }

    void insert(Alloc& alloc, Object* object, ExternalRef* ext_ref)
    {
      auto unique =
        external_map->insert(alloc, std::make_pair(object, ext_ref)).first;
      assert(unique);
      UNUSED(unique);
    }

    void erase(Alloc& alloc, Object* p)
    {
      auto it = external_map->find(p);
      assert(it != external_map->end());
      remove_ref(alloc, it);
    }

    void remove_ref(Alloc& alloc, ExternalMap::Iterator& it)
    {
      auto*& ext_ref = it.value();
      if (ext_ref != nullptr)
      {
        // The object this external ref points to has been collected, so we
        // need to invalidate this ext_ref so that `is_in` returns
        // false.second.
        ext_ref->o = nullptr;
        ext_ref->ert.store(nullptr, std::memory_order_relaxed);
        Immutable::release(alloc, ext_ref);
      }
      external_map->erase(it);
    }
  };

  using ExternalRef = ExternalReferenceTable::ExternalRef;
} // namespace verona::rt
