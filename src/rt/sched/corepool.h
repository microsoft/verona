#pragma once

#include "core.h"

#include <vector>

namespace verona::rt
{
  template<class P, class T>
  class CorePool
  {
    private:
      friend P;

      inline static Singleton<Topology, &Topology::init> topology;
      Core<T>* first_core = nullptr;
      size_t core_count = 0;
      std::vector<Core<T>*> cores;
    
    public:
      CorePool(size_t count) : core_count{count}
      {
        first_core = new Core<T>;
        Core<T>* t = first_core;
        cores.emplace_back(t);
       
        while (true)
        {
          t->affinity = topology.get().get(count);
          if (count > 1) 
          {
            t->next = new Core<T>;
            t = t->next;
            count--;
            cores.emplace_back(t);
          }
          else
          {
            t->next = first_core;
            break;
          }
        }
      }

      ~CorePool()
      {
        core_count = 0;
        first_core = nullptr;
        for (auto c: cores)
        {
          delete c;
        }
        cores.clear();
      }
  };
}
