#pragma once

#include "ds/dllist.h"
#include "threadsync.h"

//#include <mutex>

namespace verona::rt
{
  // SchedulerList implements two lists, active and free, both protected by the same lock.
  // Implementing this here rather than inside the ThreadPool decouples its logic
  // from the rest of the runtime implementation and should allow to easily change this class.
  template<class T>
  class SchedulerList
  {
    public:
      //std::mutex m;
      SchedulerLock m;

      DLList<T> active;
      DLList<T> free;

    public:
      SchedulerList() {}
      ~SchedulerList()
      {
        m.lock();
        if (!active.is_empty() || !free.is_empty())
        {
          //TODO should we abort or delete here?
          abort();
        }
        m.unlock();
      }

      void addActive(T* thread)
      {
        add(true, thread);
      }

      void addFree(T* thread)
      {
        add(false, thread); 
      }

      T* popActive()
      {
        return pop_or_null(true);
      }

      T* popFree()
      {
        return pop_or_null(false);
      }

      void moveActiveToFree(T* thread)
      {
        m.lock();
        active.remove(thread);
        free.insert_back(thread);
        m.unlock();
      }


      // WARNING: do not call other methods on the list or we end up with deadlock.
      void forall(void (*f)(T* elem))
      {
        m.lock();
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
        m.unlock();
      }

    private:
      void add(bool is_active, T* elem)
      {
        if (elem == nullptr)
          abort();
        m.lock();
        if (is_active)
          active.insert_back(elem);
        else
          free.insert_back(elem);
        m.unlock();
      }

      T* pop_or_null(bool is_active)
      {
        m.lock();
        if (is_active)
        {
          if (active.is_empty())
          {
            m.unlock();
            return nullptr;
          }
          T* res = active.pop();
          m.unlock();
          return res;
        }

        if (free.is_empty())
        {
          m.unlock();
          return nullptr;
        }
        T* res = free.pop();
        m.unlock();
        return res;
      }
  };
}
