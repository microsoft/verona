// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../object/object.h"
#include "ds/hashmap.h"
#include "externalreference.h"
#include "immutable.h"

#include <snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;

  class RememberedSet
  {
    friend class RegionTrace;
    friend class RegionArena;

  private:
    // It's not allowed to pass lambda as template argument (until c++20);
    // using static function as a workaround for HashSet and ExternalMap.

    struct HashSetEntry
    {
      Object* o;

      explicit HashSetEntry(Object* o_) : o{o_}
      {
        assert(o_);
      }

      HashSetEntry(const HashSetEntry&) = delete;

      HashSetEntry& operator=(const HashSetEntry&) = delete;

      HashSetEntry(HashSetEntry&& other) noexcept : o{other.o}
      {
        if (this != &other)
        {
          other.o = nullptr;
        }
      }

      HashSetEntry& operator=(HashSetEntry&& other) noexcept
      {
        if (this != &other)
        {
          o = other.o;
          other.o = nullptr;
        }
        return *this;
      }

      ~HashSetEntry()
      {
        if (o != nullptr)
        {
          o = HashSet::get_unmarked_pointer((size_t)o);
          RememberedSet::release_internal(ThreadAlloc::get(), o);
        }
      }
    };

    static size_t& hash_set_key_of(HashSetEntry* e)
    {
      return (size_t&)e->o;
    }
    using HashSet = PtrKeyHashMap<HashSetEntry, hash_set_key_of>;
    HashSet* hash_set;

  public:
    RememberedSet()
    {
      hash_set = HashSet::create();
    }

    inline void dealloc(Alloc* alloc)
    {
      hash_set->dealloc<false>(alloc);
      alloc->dealloc<sizeof(HashSet)>(hash_set);
    }

    void merge(Alloc* alloc, RememberedSet* that)
    {
      for (auto& e : *that->hash_set)
      {
        Object* q = HashSet::get_unmarked_pointer((size_t)e.o);
        size_t dummy;

        HashSetEntry entry{q};
        // If q is already present in this, decref, otherwise insert.
        // No need to call release, as the rc will not drop to zero.
        if (!hash_set->insert(alloc, entry, dummy))
        {
          q->decref();
        }

        // If we don't null this out, the destructor will release q.
        entry.o = nullptr;
      }
    }

    template<TransferOwnership transfer>
    void insert(Alloc* alloc, Object* o)
    {
      // If o is not present, add it and o->incref().
      assert(o->debug_is_rc() || o->debug_is_cown());

      size_t dummy;

      HashSetEntry entry{o};
      if (hash_set->insert(alloc, entry, dummy))
      {
        assert(entry.o == nullptr);
        // If the caller is not transfering ownership of a refcount, i.e., the
        // object is being added to the region but not dropped from somewhere,
        // we need to incref it.
        if constexpr (transfer == NoTransfer)
          o->incref();
      }
      else
      {
        entry.o = nullptr;
        // If the caller is transfering ownership of a refcount, i.e., the
        // object is being moved from somewhere to this region, but the object
        // is already here, we need to decref it.
        if constexpr (transfer == YesTransfer)
          o->decref();
      }
    }

    void mark(Alloc* alloc, Object* o, size_t& marked)
    {
      // If o isn't present, insert it and incref.
      assert(o->debug_is_rc() || o->debug_is_cown());

      size_t index = 0;

      HashSetEntry entry{o};
      if (hash_set->insert(alloc, entry, index))
      {
        assert(entry.o == nullptr);
        o->incref();
      }
      else
      {
        entry.o = nullptr;
      }

      hash_set->mark_slot(index, marked);
    }

    void discard(Alloc* alloc)
    {
      hash_set->clear(alloc);
    }

  private:
    static void release_internal(Alloc* alloc, Object* o)
    {
      switch (o->get_class())
      {
        case Object::RC:
        {
          assert(o->debug_is_immutable());
          Systematic::cout() << "RS releasing: immutable: " << o << std::endl;
          Immutable::release(alloc, o);
          break;
        }

        case Object::COWN:
        {
          Systematic::cout() << "RS releasing: cown: " << o << std::endl;
          cown::release(alloc, (Cown*)o);
          break;
        }

        default:
          abort();
      }
    }
  };
} // namespace verona::rt
