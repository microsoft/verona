#pragma once

#include <list>
#include <mutex>

namespace verona::rt
{
  // ThreadList is just a wrapper around a regular c++ std::list with a lock
  // for accesses.
  // This allows to change the internal datastructure (e.g., use vectors instead),
  // or later replace it completely with an in-house implementation;
  template<class T>
  class ThreadList
  {
    public:
      std::mutex m;
      std::list<T*> list;

      ThreadList() {}
      ~ThreadList()
      {
        m.lock();
        if (!list.empty())
        {
          //TODO should we abort or delete here?
          abort();
        }
        m.unlock();
      }

      void push_back(T* thread)
      {
        m.lock();
        list.push_back(thread);
        m.unlock();
      }

      T* pop_front_or_null()
      {
        m.lock();
        if (list.empty())
        {
          m.unlock();
          return nullptr;
        }
        T* res = list.front();
        list.pop_front();
        m.unlock();
        return res;
      }

      void remove(T* thread)
      {
        m.lock();
        list.remove(thread);
        m.unlock();
      }

      // WARNING: do not call other methods on the list or we end up with deadlock.
      void forall(void (*f)(T* elem))
      {
        m.lock();
        for (auto t: list)
        {
          f(t);
        }
        m.unlock();
      }
  };

}
