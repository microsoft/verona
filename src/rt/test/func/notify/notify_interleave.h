// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
namespace notify_interleave
{
  struct Ping : public VBehaviour<Ping>
  {
    void f() {}
  };

  bool g_called = false;

  struct A : public VCown<A>
  {
    void notified(Object* o)
    {
      auto a = (A*)o;
      (void)a;
      g_called = true;
    }
  };

  enum Phase
  {
    NOTIFYSEND,
    WAIT,
    EXIT,
  };

  struct B : public VCown<B>
  {
    A* a;
    // Here we wait for 100 msgs so that we know a has been scheduled.
    int wait_count = 100;
    Phase state = NOTIFYSEND;

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
        case NOTIFYSEND:
        {
          g_called = false;
          a->mark_notify();
          Cown::schedule<Ping>(a);
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
            b->state = EXIT;
          }
          Cown::schedule<Loop>(b, b);
          break;
        }

        case EXIT:
        {
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

  // This test confirms that `mark_notify` info is preserved when new messages
  // are sent to that cown.
  void run_test()
  {
    auto a = new A;
    auto b = new B(a);

    Cown::schedule<Loop>(b, b);
  }
}
