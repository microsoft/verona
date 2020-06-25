// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

namespace ext_ref_basic
{
  struct Node;
  struct DList;
  struct A;
  struct B;

  A* g_a;
  DList* g_list;

  struct Node : public V<Node>
  {
    static int count;
    int id;
    B* element;
    Node* prev = nullptr;
    Node* next = nullptr;

    Node(B* b = nullptr) : element{b}
    {
      id = count++;
    }
  };

  int Node::count = -2;

  struct B : VCown<B>
  {
    ExternalRef* ext_node;

    ExternalRef* ext_node_alias;

    B(Object* list, Object* node)
    {
      auto region = Region::get(list);
      ext_node = ExternalRef::create(region, node);
      ext_node_alias = ExternalRef::create(region, node);
    }

    void trace(ObjectStack& st) const
    {
      assert(ext_node);
      st.push(ext_node);
      assert(ext_node_alias);
      st.push(ext_node_alias);
    }
  };

  struct DList : public V<DList>
  {
    Node* first;
    Node* last;

    DList(int n)
    {
      first = new (this) Node;
      last = new (this) Node;

      first->next = last;
      last->prev = first;

      auto* alloc = ThreadAlloc::get();
      auto cur = first;
      for (auto i = 0; i < n; ++i)
      {
        auto node = new (this) Node;

        auto b = new B(this, node);
        node->element = b;
        RegionTrace::insert<YesTransfer>(alloc, this, b);

        cur->next = node;
        cur->next->prev = cur;
        cur = cur->next;
      }
      cur->next = last;
      last->prev = cur;
    }

    void print()
    {
      auto cur = first->next;
      while (cur != last)
      {
        printf("%d, ", cur->id);
        cur = cur->next;
      }
      printf("\n");
    }
  };

  struct A : VCown<A>
  {
    DList* list;

    A(DList* list_) : list{list_} {}

    void trace(ObjectStack& st) const
    {
      assert(list);
      st.push(list);
    }
  };

  struct AMsg : public VAction<AMsg>
  {
    A* a;
    B* b;
    ExternalRef* ext_node;

    Node* get()
    {
      return (Node*)ext_node->get();
    }

    void f()
    {
      auto alloc = ThreadAlloc::get();

      auto list = a->list;
      assert(ext_node->is_in(Region::get(g_list)));
      auto node = get();

      node->prev->next = node->next;
      node->next->prev = node->prev;

      node->element = nullptr;
      RegionTrace::gc(alloc, list);

      assert(!ext_node->is_in(Region::get(g_list)));
      Immutable::release(alloc, ext_node);
    }

    AMsg(A* a, B* b_, ExternalRef* ext_node_) : a(a), b{b_}, ext_node{ext_node_}
    {}
  };

  // Illustrating a scenario, where cown A holds a doubly-linked list, with
  // each node holds a ref to cown B. When cown B decides to remove itself
  // from this list, it could send an message to cown A. Upon receiving this
  // message, cown A will find the corresponding node in constant time via
  // this external ref, and unlink the corresponding node. The use of schedulers
  // is merely for illustration purpose.
  void basic_test()
  {
    Scheduler& sched = Scheduler::get();
    size_t cores = 1;
    sched.init(cores);

    auto* alloc = ThreadAlloc::get();
    (void)alloc;

    g_list = new DList(1);
    g_a = new A(g_list);

    auto b = g_list->first->next->element;

    // Aliasing ext_node in AMsg.
    Immutable::acquire(b->ext_node);
    Cown::schedule<AMsg>(g_a, g_a, b, b->ext_node);

    Cown::release(alloc, g_a);
    sched.run();
    snmalloc::current_alloc_pool()->debug_check_empty();
  }

  template<RegionType region_type>
  struct R : public V<R<region_type>, region_type>
  {};

  template<RegionType region_type>
  void singleton_region_test()
  {
    auto* alloc = ThreadAlloc::get();
    (void)alloc;

    auto r = new (alloc) R<region_type>;
    auto region = Region::get(r);
    auto ext_ref = ExternalRef::create(region, r);
    assert(ext_ref->is_in(region));
    assert(ext_ref->get() == r);

    Immutable::release(alloc, ext_ref);
    Region::release(alloc, r);

    snmalloc::current_alloc_pool()->debug_check_empty();
  }

  void run_test()
  {
    basic_test();
    singleton_region_test<RegionType::Trace>();
    singleton_region_test<RegionType::Arena>();
  }

  void run_all()
  {
    run_test();
  }
}
