// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include <test/harness.h>
#include <verona.h>

using namespace snmalloc;
using namespace verona::rt;

namespace ext_ref_basic
{
  struct Node;
  struct DList;
  struct A;
  struct B;

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
      UsingRegion ur(list);
      ext_node = create_external_reference(node);
      ext_node_alias = create_external_reference(node);
    }

    void trace(ObjectStack& st) const
    {
      check(ext_node);
      st.push(ext_node);
      check(ext_node_alias);
      st.push(ext_node_alias);
    }
  };

  struct DList : public V<DList>
  {
    Node* first;
    Node* last;

    DList(int n)
    {
      UsingRegion ur(this);
      first = new Node;
      last = new Node;

      first->next = last;
      last->prev = first;

      auto& alloc = ThreadAlloc::get();
      auto cur = first;
      for (auto i = 0; i < n; ++i)
      {
        auto node = new Node;

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
      check(list);
      st.push(list);
    }
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

    auto& alloc = ThreadAlloc::get();
    (void)alloc;

    auto list = new (RegionType::Trace) DList(1);
    auto a = new A(list);

    auto b = list->first->next->element;

    // Aliasing ext_node in Closure.
    Immutable::acquire(b->ext_node);
    auto ext_node = b->ext_node;
    schedule_lambda(a, [a, ext_node]() {
      auto& alloc = ThreadAlloc::get();

      auto list = a->list;
      UsingRegion ur(list);
      check(is_external_reference_valid(ext_node));
      auto node = (Node*)use_external_reference(ext_node);

      node->prev->next = node->next;
      node->next->prev = node->prev;

      node->element = nullptr;
      region_collect();

      check(!is_external_reference_valid(ext_node));
      Immutable::release(alloc, ext_node);
    });
    Cown::release(alloc, a);
    sched.run();
    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  struct R : public V<R>
  {};

  template<RegionType region_type>
  void singleton_region_test()
  {
    auto& alloc = ThreadAlloc::get();
    (void)alloc;

    auto r = new (region_type) R;
    {
      UsingRegion ur(r);
      auto ext_ref = create_external_reference(r);
      check(is_external_reference_valid(ext_ref));
      check(use_external_reference(ext_ref) == r);
      Immutable::release(alloc, ext_ref);
    }

    region_release(r);

    snmalloc::debug_check_empty<snmalloc::Alloc::StateHandle>();
  }

  void run_test()
  {
    basic_test();
    singleton_region_test<RegionType::Trace>();
    singleton_region_test<RegionType::Arena>();
    // TODO: RegionType::Rc
  }

  void run_all()
  {
    run_test();
  }
}
