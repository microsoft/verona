// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "../object/object.h"
#include "linked_object_stack.h"

namespace verona::rt
{
  namespace cown
  {
    // This is used only to break a dependency cycle.
    inline void release(Alloc* alloc, Cown* o);
    inline void mark_for_scan(Object* o, EpochMark epoch);
  } // namespace cown

  class Immutable
  {
  public:
    static void acquire(Object* o)
    {
      assert(o->debug_is_immutable());
      o->immutable()->incref();
    }

    static size_t release(Alloc* alloc, Object* o)
    {
      assert(o->debug_is_immutable());
      auto root = o->immutable();

      if (root->decref())
        return free(alloc, root);

      return 0;
    }

    static void mark_and_scan(Alloc* alloc, Object* o, EpochMark epoch)
    {
      assert(o->debug_is_immutable());
      auto root = o->immutable();

      ObjectStack dfs(alloc);
      dfs.push(root);

      while (!dfs.empty())
      {
        o = dfs.pop();

        switch (o->get_class())
        {
          case Object::RC:
          case Object::SCC_PTR:
          {
            Systematic::cout()
              << "Immutable Scan: reaches immutable: " << o << std::endl;
            if (o->in_epoch(epoch))
              continue;

            // This may trace an immutable that has already been traced, as it
            // races over the epoch mark. This is ok.
            o->set_epoch(epoch);
            o->trace(dfs);
            break;
          }

          case Object::COWN:
          {
            Systematic::cout()
              << "Immutable Scan: reaches cown: " << o << std::endl;
            cown::mark_for_scan(o, epoch);
            break;
          }

          default:
            abort();
        }
      }
    }

  private:
    static size_t free(Alloc* alloc, Object* o)
    {
      assert(o == o->immutable());
      size_t total = 0;

      // Free immutable graph.
      ObjectStack f(alloc);
      LinkedObjectStack fl;
      LinkedObjectStack scc;
      LinkedObjectStack dfs;

      dfs.push(o);

      while (!dfs.empty())
      {
        assert(f.empty());
        assert(fl.empty());
        assert(scc.empty());

        Object* v = dfs.pop();
        v->trace(f);

        while (!f.empty())
        {
          Object* w = f.pop();
          scc_classify(alloc, w, dfs, scc);
        }

        while (!scc.empty())
        {
          Object* w = scc.pop();
          fl.push(w);
          w->trace(f);

          while (!f.empty())
          {
            Object* u = f.pop();
            scc_classify(alloc, u, dfs, scc);
          }
        }

        // Run all finalisers for this SCC before deallocating.
        fl.forall<run_finaliser>();

        // We don't need the actual subregions here, as they have been frozen.
        ObjectStack dummy(alloc);
        v->finalise(nullptr, dummy);

        while (!fl.empty())
        {
          Object* w = fl.pop();
          total += w->size();
          w->destructor();
          w->dealloc(alloc);
        }

        total += v->size();
        v->destructor();
        v->dealloc(alloc);
      }

      assert(f.empty());
      assert(fl.empty());
      assert(scc.empty());
      assert(dfs.empty());

      return total;
    }

    static inline void run_finaliser(Object* o)
    {
      // We don't need the actual subregions here, as they have been frozen.
      ObjectStack dummy(ThreadAlloc::get_noncachable());
      o->finalise(nullptr, dummy);
    }

    static inline void scc_classify(
      Alloc* alloc, Object* w, LinkedObjectStack& dfs, LinkedObjectStack& scc)
    {
      Object::RegionMD c;
      Object* r = w->root_and_class(c);

      switch (c)
      {
        case Object::RC:
        {
          if (r->decref())
            dfs.push(r);
          break;
        }

        case Object::UNMARKED:
        {
          if (w != r)
          {
            scc.push(w);
          }
          break;
        }

        case Object::COWN:
        {
          Systematic::cout() << "Immutable releasing cown: " << w << std::endl;
          cown::release(alloc, (Cown*)w);
          break;
        }

        default:
          assert(0);
      }
    }
  };
} // namespace verona::rt
