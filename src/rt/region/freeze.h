// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "region.h"

namespace verona::rt
{
  /**!freeze.md
   * Freezing an object graph
   * ========================
   *
   * The Freeze class provides the methods for freezing an isolated object
   * graph.
   *
   * In does two key things
   *  - calculates the union/find data structure for strongly connected
   *    components (SCC)
   *  - calculates the incoming references to an SCC. Pointers inside the SCC
   *    are not counted.
   *
   * Abstract Algorithm
   * ------------------
   *
   * We first explain the algorithm ignoring
   *
   *   - Not blowing up the stack, i.e. we give a recursive algorithm initially
   *   - nested isolated regions
   *   - pointers out of the isolated region into already immutable state or
   *     Cowns
   *   - optimisations to not use atomic reference counting while construction
   *
   * We use a union-find structure in the RegionMD field. The RegionMD field is
   * one of
   *
   *   - `UNMARKED`    - not yet visited
   *   - `SCC_PTR(p)`  - a pointer to another object that either directly or
   *                     indirectly knows its status
   *   - `RC(N)`       - A complete SCC with refcount N
   *   - `PENDING(N)`  - A incomplete SCC where the union-find structure has at
   *                   most depth N chains of SCC_PTR.
   *
   * In the abstract algorithm below, we define rep to find the root in the
   * union-find datastructure
   *
   * The core idea follows most path-based SCC algorithms. We use pending to
   * represent the current path. It is a stack of SCCs from the root to where we
   * are currently working. Foreach SCC on the current path, the entries are the
   * node highest in the spanning tree that DFS is walking. These entries are
   * not neccessarily the representatives in the union find structure. This is
   * important to allow the union-find structure to remain balanced.

   * Here is the abstract algorithm
   *
   * ```
   *   freeze (r) =
   *     pending = empty_stack
   *
   *     let rec freeze_inner(x) =
   *       match rep(x).status with
   *       | UNMARKED =>
   *         x.set_pending();
   *         pending.push(x);
   *         for each f in x
   *           freeze_inner(x.f)
   *         if (pending.peek() == x)         // (A) Post-order check
   *           x."makescc with ref count 1"()
   *
   *       | PENDING(N) =>
   *         while (rep(x) != rep(pending.peek()))
   *           union(x, pending.pop())
   *
   *       | RC(N) =>
   *           x.status = RC(N+1)
   *
   *     freeze_inner (r)
   *
   *     forall o : object
   *       if (o.status ==  UNMARKED)
   *          finalise o
   *
   *     forall o : object
   *       if (o.status ==  UNMARKED)
   *          deallocate o
   * ```
   *
   * While walking the DFS, if we encounter a `PENDING` node, then we can
   * collapse everything on the path to the back edge.  If we encounter a `RC`,
   * then we have found a reference into a complete SCC, and should up its
   * reference count.
   *
   * Implemenation
   * -------------
   *
   * In the runtime code, we use a dfs stack, rather than the actual call stack.
   * To achieve the postorder check (A), we use the bottom bit in the DFS entry
   * stack to represent that this is a postorder access.
   *
   * In the code we use:
   *
   *  * `UNMARKED` for a node we have not yet visited
   *  * `PENDING` represents the current path from the root to where we are
   *    working.  The value stored in pending represents the maximum depth of
   *    SCC chain.
   *  * `RC_NON_ATOMIC` represents a completed SCC root.
   *    The value stored represents the number of incoming edges found so far.
   *  * `SCC_PTR` represents something that has been found to be part of a
   *    cycle. If, via a series of zero or more `SCC_PTR`, this points to
   *    `PENDING` then this is part of an SCC that is still being calculated.
   *    If via a series of zero of more SCC_PTR this points to an
   *    `RC_NON_ATOMIC`, then this is part of a completed SCC.
   *  * `Object::RC` and `Object::COWN` refer to immutable objects outside the
   *    current isolate.
   *
   * The objects stack is used to keep track of the set of objects in the
   * region. Rather than copy the set up front, we lazily construct it using the
   * ring in the isolated regions. Every time we break the ring, we keep track
   * of that point in the objects stack.
   */
  class Freeze
  {
  private:
    static Object* post_order_mark(Object* o)
    {
      return (Object*)(((size_t)o) | 1);
    }

    static Object* remove_post_order_mark(Object* o)
    {
      return (Object*)(((uintptr_t)o) & ~(uintptr_t)1);
    }

  public:
    static void apply(Alloc& alloc, Object* o)
    {
      assert(o->debug_is_iso());

      ObjectStack objects(alloc);
      ObjectStack dfs(alloc);
      ObjectStack iso(alloc);
      ObjectStack pending(alloc);
      ObjectStack dealloc_regions(alloc);

      iso.push(o);

      while (!iso.empty())
      {
        assert(objects.empty());
        assert(dfs.empty());

        Object* p = iso.pop();
        assert(p->debug_is_iso());

        // TODO(region): Right now we can only freeze trace regions. We'll
        // probably need different strategies if we want to freeze other kinds
        // of regions, e.g. copying objects out of an arena region.
        assert(RegionTrace::is_trace_region(p->get_region()));
        RegionTrace* reg = RegionTrace::get(p);

        // Drop the ISO mark on the entry point.
        p->init_next(reg);

        // Start with the graph entry point.
        dfs.push(p);

        // Add the finaliser, and non-finaliser rings to objects.
        objects.push(reg->next_not_root);
        objects.push(reg->get_next());

        // Mark region metadata object, so sweeping does not travel through it.
        reg->Object::mark();

        while (!dfs.empty())
        {
          Object::RegionMD c;

          // Depth-first search has reached vertex q.
          // This may be either a pre-order and post-order visit
          Object* q_mark = dfs.pop();
          Object* q = remove_post_order_mark(q_mark);

          if (q != q_mark)
          {
            // Finished this part of the spanning tree
            // If this is the head of the pending list, this means we have
            // processed all children in the spanning tree and this should now
            // be turned into a complete SCC with ref count 1.
            if (q == pending.peek())
            {
              pending.pop();
              q->root_and_class(c)->make_nonatomic_scc();
              assert(c == Object::PENDING);
            }
            continue;
          }

          auto r = q->root_and_class(c);

          switch (c)
          {
            case Object::PENDING:
            {
              // We have found a reference back into one of the SCCs
              // on the current path.  Collapse the path by unioning
              // all the nodes up to that SCC.
              auto rank = r->pending_rank();
              while (r != (p = pending.peek()->root_and_class(c)))
              {
                assert(c == Object::PENDING);
                // Rank used to keep the union/find data structure balanced
                auto p_rank = p->pending_rank();
                if (p_rank <= rank)
                {
                  p->set_scc(r);
                  if (p_rank == rank)
                    r->set_pending_rank(++rank);
                }
                else
                {
                  r->set_scc(p);
                  rank = p_rank;
                  r = p;
                }
                pending.pop();
              }
              break;
            }

            case Object::ISO:
            {
              // External Iso, process that later.
              iso.push(q);
              break;
            }

            case Object::RC:
            case Object::COWN:
            {
              Logging::cout()
                << "External reference during freeze: " << r << Logging::endl;
              // External reference
              r->incref();
              break;
            }

            case Object::NONATOMIC_RC:
            {
              // Reference to an already complete SCC, so incref it.
              r->incref_nonatomic();
              break;
            }

            case Object::UNMARKED:
            {
              // Lazily construct stack of sublists for gcing
              objects.push(q->get_next());
              // Clear the `has_ext_ref` bit.
              q->clear_has_ext_ref();
              // Add this to the current path we are exploring
              q->set_pending();
              pending.push(q);
              // Push post-order mark, so we can revisit once subtree complete
              dfs.push(post_order_mark(q));
              // Add all the fields to the dfs
              q->trace(dfs);
              break;
            }

            default:
              assert(0);
          }
        }

        // Finalise all the objects
        // Move non-atomics to atomics
        // Calculate list of things to be deallocated
        LinkedObjectStack to_dealloc;
        p = objects.pop();
        while (true)
        {
          switch (p->get_class())
          {
            case Object::UNMARKED:
            {
              // Node was unreachable deallocate it
              auto next = p->get_next();

              assert(p != reg);

              // ISO marker has been dropped on entry point, so
              // can pass nullptr here.
              p->finalise(nullptr, dealloc_regions);
              to_dealloc.push(p);
              // Deallocate unreachable sub-regions
              while (!dealloc_regions.empty())
              {
                Object* q = dealloc_regions.pop();
                Region::release(alloc, q);
              }

              p = next;
              continue;
            }

            case Object::NONATOMIC_RC:
            {
              // Convert to atomic rc to allow sharing.
              p->make_atomic();
              break;
            }

            case Object::MARKED:
              assert(p == reg);

            case Object::RC:
            case Object::SCC_PTR:
              break;

            default:
              assert(0);
          }

          if (objects.empty())
            break;

          p = objects.pop();
        }

        // Finally deallocate objects.
        while (!to_dealloc.empty())
        {
          Object* q = to_dealloc.pop();
          q->destructor();
          q->dealloc(alloc);
        }

        reg->discard(alloc);
        reg->dealloc(alloc);
      }

      assert(objects.empty());
      assert(dfs.empty());
      assert(iso.empty());
    }
  };
} // namespace verona::rt
