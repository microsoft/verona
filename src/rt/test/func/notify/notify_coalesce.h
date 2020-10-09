// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
namespace notify_coalesce
{
  static size_t g_called;

  struct A : public VCown<A>
  {
    void notified(Object*)
    {
      Systematic::cout() << "Recv Notify" << std::endl;
      g_called++;
    }
  };

  struct Ping : public VBehaviour<Ping>
  {
    void f()
    {
      Systematic::cout() << "Recv Ping" << std::endl;
    }
  };

  enum Phase
  {
    NOTIFY,
    WAIT,
    CHECK,
  };

  struct B : public VCown<B>
  {
    A* a;
    // Here we wait for 100 msgs so that we know `a` has been scheduled.
    int wait_count = 100;
    Phase state = NOTIFY;

    B(A* a_) : a{a_} {}

    void trace(ObjectStack& st) const
    {
      check(a);
      st.push(a);
    }
  };

  struct Loop : public VBehaviour<Loop>
  {
    B* b;
    Loop(B* b) : b(b) {}

    void f()
    {
      auto a = b->a;
      switch (b->state)
      {
        case NOTIFY:
        {
          g_called = 0;
          for (int i = 0; i < 10; ++i)
          {
            Systematic::cout() << "Send Notify" << std::endl;
            a->mark_notify();
            Cown::schedule<Ping>(a);
          }
          b->state = WAIT;
          Cown::schedule<Loop>(b, b);
          break;
        }

        case WAIT:
        {
          if (b->wait_count > 0)
          {
            b->wait_count--;
          }
          else
          {
            b->state = CHECK;
          }
          Cown::schedule<Loop>(b, b);
          break;
        }

        case CHECK:
        {
          // If this check fails, check if BATCH_COUNT is lower than 100.
          check(g_called == 1);
          Cown::release(ThreadAlloc::get(), b);
          break;
        }

        default:
        {
          abort();
        }
      }
    }
  };

  // This test verifies that multiple calls to `mark_notify` could be
  // coalesced, causing single acknowledge on the cown.
  void run_test()
  {
    auto* alloc = ThreadAlloc::get();

    auto a = new A;
    auto b = new B(a);

    auto a2 = new A;
    Cown::schedule<Ping>(a);
    Cown::schedule<Loop>(b, b);
    Cown::release(alloc, a2);
  }
}
