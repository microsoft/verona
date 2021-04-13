// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ds/hashmap.h"
#include "object/object.h"

#include <snmalloc.h>

namespace verona::rt
{
  using namespace snmalloc;

  class Pollers
  {
    ObjectMap<Object*> set;
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    size_t select_index = 0;

  public:
    Pollers() : set{ThreadAlloc::get_noncachable()} {}

    void add_poller(Object* poller)
    {
      auto* alloc = ThreadAlloc::get_noncachable();
      FlagLock l{lock};
      set.insert(alloc, poller);
    }

    void remove_poller(const Object* poller)
    {
      FlagLock l{lock};
      set.erase(poller);
    }

    size_t count()
    {
      FlagLock l{lock};
      return set.size();
    }

    Object* select()
    {
      FlagLock l{lock};

      if (select_index >= set.size())
        select_index = 0;

      size_t count = 0;
      Object* ret = nullptr;
      for (auto* it : set)
      {
        if (count++ == select_index)
        {
          ret = it;
          break;
        }
      }
      select_index = count;
      return ret;
    }
  };
}
