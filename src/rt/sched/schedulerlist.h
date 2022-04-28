#pragma once

#include <list>
#include <mutex>

namespace verona::rt
{
  // SchedulerList implements two lists, active and free, both protected by the same lock.
  // Implementing this here rather than inside the ThreadPool decouples its logic
  // from the rest of the runtime implementation and should allow to easily change this class.
  template<class T>
  class SchedulerList
  {
    public:
      std::mutex m;

      std::list<T*> active;
      std::list<T*> free;

    public:
      SchedulerList() {}
      ~SchedulerList()
      {
        m.lock();
        if (!active.empty() || !free.empty())
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
        free.push_back(thread);
        m.unlock();
      }


      // WARNING: do not call other methods on the list or we end up with deadlock.
      void forall(void (*f)(T* elem))
      {
        m.lock();
        for (auto t: active)
        {
          f(t);
        }
        for (auto t: free)
        {
          f(t);
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
          active.push_back(elem);
        else
          free.push_back(elem);
        m.unlock();
      }

      T* pop_or_null(bool is_active)
      {
        m.lock();
        std::list<T*>* list = &free;
        if (is_active)
        {
          list = &active;
        }

        if (list->empty())
        {
          m.unlock();
          return nullptr;
        }
        T* res = list->front();
        list->pop_front();
        m.unlock();
        return res;
      }
  };

}
