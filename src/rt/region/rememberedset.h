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
    struct SetCallbacks
    {
      static void on_insert(Object*&) {}

      static void on_erase(Object*& e)
      {
        if (e != nullptr)
          RememberedSet::release_internal(
            ThreadAlloc::get(), HashSet::unmark_pointer(e));
      }
    };

    using HashSet = PtrKeyHashMap<Object*, SetCallbacks>;
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
      for (auto*& e : *that->hash_set)
      {
        Object* q = HashSet::unmark_pointer(e);

        // If q is already present in this, decref, otherwise insert.
        // No need to call release, as the rc will not drop to zero.
        if (!hash_set->insert(alloc, q))
        {
          q->decref();
        }
      }
    }

    template<TransferOwnership transfer>
    void insert(Alloc* alloc, Object* o)
    {
      // If o is not present, add it and o->incref().
      assert(o->debug_is_rc() || o->debug_is_cown());

      // If the caller is not transfering ownership of a refcount, i.e., the
      // object is being added to the region but not dropped from somewhere,
      // we need to incref it.
      if constexpr (transfer == NoTransfer)
        o->incref();

      if (!hash_set->insert(alloc, o))
      {
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
      if (hash_set->insert(alloc, o, index))
        o->incref();

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
