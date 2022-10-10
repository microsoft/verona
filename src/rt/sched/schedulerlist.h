// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ds/dllist.h"
#include "threadsync.h"

#include <cstdlib>
#include <snmalloc/snmalloc.h>

namespace verona::rt
{
  // SchedulerList implements two lists, active and free, both protected by the
  // same lock. Implementing this here rather than inside the ThreadPool
  // decouples its logic from the rest of the runtime implementation and should
  // allow to easily change this class.
  template<class T>
  class SchedulerList
  {
  public:
    snmalloc::FlagWord m;

    DLList<T> active;
    DLList<T> free;

  public:
    constexpr SchedulerList() = default;

#ifndef NDEBUG
    ~SchedulerList()
    {
      snmalloc::FlagLock lock(m);
      if (!active.is_empty() || !free.is_empty())
      {
        abort();
      }
    }
#endif

    void add_active(T* thread)
    {
      add(active, thread);
    }

    void add_free(T* thread)
    {
      add(free, thread);
    }

    T* pop_active()
    {
      return pop_or_null(active);
    }

    T* pop_free()
    {
      return pop_or_null(free);
    }

    void move_active_to_free(T* thread)
    {
      snmalloc::FlagLock lock(m);
      active.remove(thread);
      free.insert_back(thread);
    }

    // Applies f on all active and free threads.
    // WARNING: do not call other methods on the list or we end up with
    // deadlock.
    void forall(void (*f)(T* elem))
    {
      snmalloc::FlagLock lock(m);
      T* t = active.get_head();
      while (t != nullptr)
      {
        f(t);
        t = t->next;
      }
      t = free.get_head();
      while (t != nullptr)
      {
        f(t);
        t = t->next;
      }
    }

    void dealloc_lists()
    {
      dealloc_active();
      dealloc_free();
    }

    // Empty the active list and call delete on every element.
    void dealloc_active()
    {
      dealloc_list(active);
    }

    // Empty the free list and call delete on every element.
    void dealloc_free()
    {
      dealloc_list(free);
    }

  private:
    void add(DLList<T>& list, T* elem)
    {
      if (elem == nullptr)
        abort();
      snmalloc::FlagLock lock(m);
      list.insert_back(elem);
    }

    T* pop_or_null(DLList<T>& list)
    {
      snmalloc::FlagLock lock(m);
      if (list.is_empty())
        return nullptr;
      return list.pop();
    }

    void dealloc_list(DLList<T>& list)
    {
      T* t = list.pop();
      while (t != nullptr)
      {
        T* prev = t;
        t = list.pop();
        delete prev;
      }
    }
  };
}
