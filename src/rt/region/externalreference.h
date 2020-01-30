// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../object/object.h"
#include "ds/hashmap.h"
#include "immutable.h"

#include <snmalloc.h>

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

      static void gc_trace(const Object*, ObjectStack*) {}

      static const Descriptor* desc()
      {
        static constexpr Descriptor desc = {
          sizeof(ExternalRef), gc_trace, nullptr, nullptr};

        return &desc;
      }

      // May only be called if there's a ext_ref for o in rs.
      static ExternalRef* find_ext_ref(ExternalReferenceTable* ert, Object* o)
      {
        auto i = ert->external_map->find((size_t)o);
        assert(i != ert->external_map->end());
        return i->second.get_wref();
      }

      ExternalRef(ExternalReferenceTable* ert_, Object* o_) : ert{ert_}, o{o_}
      {
        set_descriptor(desc());
        make_scc();

        auto pair = std::make_pair((size_t)o, ExternalRefHolder{this});
        ert.load(std::memory_order_relaxed)
          ->external_map->insert_unique(ThreadAlloc::get(), pair);

        o->set_has_ext_ref();
      }

      void* operator new(size_t size)
      {
        return ThreadAlloc::get_noncachable()->alloc(size);
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

        auto ext_ref = new ExternalRef(ert, o);
        return ext_ref;
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

    /**
     * ExternalRef wrapper holding the regions ownership on the ExternalRef.
     * Destructor invalidates the ExternalRef.
     */
    class ExternalRefHolder
    {
    private:
      ExternalRef* ext_ref;

    public:
      ExternalRefHolder() : ext_ref{nullptr} {}

      explicit ExternalRefHolder(ExternalRef* ext_ref_) : ext_ref{ext_ref_}
      {
        assert(ext_ref);
        ext_ref->incref();
      }

      ExternalRefHolder(const ExternalRefHolder&) = delete;

      ExternalRefHolder& operator=(const ExternalRefHolder&) = delete;

      ExternalRefHolder(ExternalRefHolder&& other) noexcept
      : ext_ref{other.ext_ref}
      {
        if (this != &other)
        {
          other.ext_ref = nullptr;
        }
      }

      ExternalRefHolder& operator=(ExternalRefHolder&& other) noexcept
      {
        if (this != &other)
        {
          ext_ref = other.ext_ref;
          other.ext_ref = nullptr;
        }
        return *this;
      }

      void set_ert(ExternalReferenceTable* ert_)
      {
        assert(ext_ref->o);
        ext_ref->ert.store(ert_, std::memory_order_relaxed);
      }

      ExternalRef* get_wref()
      {
        assert(ext_ref);
        return ext_ref;
      }

      ~ExternalRefHolder()
      {
        if (ext_ref)
        {
          // The object this external ref points to has been collected, so we
          // need to invalidate this ext_ref so that `is_in` return false.
          ext_ref->o = nullptr;
          ext_ref->ert.store(nullptr, std::memory_order_relaxed);
          Alloc* alloc = ThreadAlloc::get();
          Immutable::release(alloc, ext_ref);
        }
      }
    };

    static size_t& external_map_key_of(std::pair<size_t, ExternalRefHolder>* e)
    {
      return e->first;
    }

    // No tracing is need for external_map, because entries in the map doesn't
    // contribute to objects RC; when an object is collected, its corresponding
    // entry in the map (if any) is removed as well.
    using ExternalMap =
      PtrKeyHashMap<std::pair<size_t, ExternalRefHolder>, external_map_key_of>;

    ExternalMap* external_map;

  public:
    ExternalReferenceTable()
    {
      external_map = ExternalMap::create();
    }

    inline void dealloc(Alloc* alloc)
    {
      external_map->dealloc(alloc);
      alloc->dealloc<sizeof(ExternalMap)>(external_map);
    }

    void merge(Alloc* alloc, ExternalReferenceTable* that)
    {
      for (auto& e : *that->external_map)
      {
        e.second.set_ert(this);
        auto pair = std::make_pair(e.first, std::move(e.second));
        external_map->insert_unique(alloc, pair);
      }
    }

    void erase(Object* p)
    {
      external_map->erase(p);
    }
  };

  using ExternalRef = ExternalReferenceTable::ExternalRef;
} // namespace verona::rt
