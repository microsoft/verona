// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This example tests the notification where the message queue is empty
 *
 * The code is performing the following psuedo Verona program:

    wait(c1, c2)
    {
        when (c1)
        {
            if (c1.count == 10)
            {
                when (c1, c2)
                {
                    print ("We recevied :" + c2.count + " notifications.");
                    assert(c2.count >= 10);
                    assert(c2.count <= 20);
                    c2.closed = true;
                }
            }
            else
                if (c1.go)
                {
                    c1.go = false;
                    c1.count++;
                    c2.notify();
                    c2.notify();
                }
                wait(c1)
        }
    }

    c2::notify()
    {
        if (c2.closed) error();

        c2.count++;
        c1 = c2.observer;
        when (c1)
        {
            c1.go = true
        }
    }
 *
 * Note that systematic testing is not finding counts very far from
 * 10 for the notified cown.
 */

namespace notify_empty_queue
{
  struct MyCown : VCown<MyCown>
  {
    size_t count = 0;
    MyCown* observer = nullptr;
    bool closed = false;
    bool go = true;

    static void notified(Object* o);
  };

  /**
   * Performs from above:

        when (c1)
        {
            c1.go = true
        }
  */
  struct Go : public VBehaviour<Go>
  {
    MyCown* leader;

    Go(MyCown* leader) : leader(leader) {}

    void f()
    {
      std::cout << "Go!" << std::endl;

      leader->go = true;
    }
  };

  /**
   * Corresponds to
        when (c1, c2)
        {
            print ("We recevied :" + c2.count + " notifications.");
            assert(c2.count >= 10);
            assert(c2.count <= 20);
            c2.closed = true;
        }
    */
  struct Finish : public VBehaviour<Finish>
  {
    MyCown* leader;
    MyCown* notified;

    Finish(MyCown* leader, MyCown* notified)
    : leader(leader), notified(notified)
    {}

    void f()
    {
      std::cout << "Received " << notified->count << " notifications."
                << std::endl;
      check(leader->count == 10);
      check(notified->count >= 10);
      check(notified->count <= 20);
      notified->closed = true;

      auto& alloc = ThreadAlloc::get();
      // The initial test set up forgot about reference counts to the two
      // cowns involved in this example.  This should be the last event in
      // the system.  Remove the reference counts to deallocate the cowns.
      Cown::release(alloc, leader);
      Cown::release(alloc, notified);
    }
  };

  /**
   * Performs the large when inside wait in the comment above.
   */
  struct Wait : public VBehaviour<Wait>
  {
    MyCown* leader;
    MyCown* notified;

    Wait(MyCown* leader, MyCown* notified) : leader(leader), notified(notified)
    {}

    void f()
    {
      if (leader->count == 10)
      {
        Cown* cowns[2] = {leader, notified};
        Cown::schedule<Finish>(2, cowns, leader, notified);
      }
      else
      {
        std::cout << "Try go!" << std::endl;
        if (leader->go)
        {
          std::cout << "Incremement count!" << std::endl;
          leader->go = false;
          leader->count++;
          notified->mark_notify();
          notified->mark_notify();
        }
        else
        {
          std::cout << "No go!" << std::endl;
        }
        Cown::schedule<Wait>(leader, leader, notified);
      }
    }
  };

  void MyCown::notified(Object* o)
  {
    std::cout << "Notification!" << std::endl;
    auto c = (MyCown*)o;
    if (c->closed)
    {
      std::cout << "All notifications should have been processed!" << std::endl;
      abort();
    }

    c->count++;
    Cown::schedule<Go>(c->observer, c->observer);
  }

  void run_test()
  {
    auto leader = new MyCown;
    auto notified = new MyCown;
    notified->observer = leader;

    Cown::schedule<Wait>(leader, leader, notified);
    // We have left two reference counts unused for leader and notified.
    // These will be dec refed in the Finish message. This saves
    // having to correctly manage the reference counts.
  }
}
