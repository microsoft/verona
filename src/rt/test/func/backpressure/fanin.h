
/* TODO
class Sender
{
  messages: USize

  create(messages: USize): cown[Sender] & imm
  {
    let sender = new cown[Sender]
    sender.messages = messages
    sender
  }

  apply(self: cown[Sender] & imm, receivers: Array[EmptyCown] & imm)
  {
    when (self)
    {
      forward(receivers)

      self.messages = self.messages - 1
      if (self.messages > 0)
      {
        self.apply()
      }
    }
  }
}

forward(receivers: Array[EmptyCown] & imm)
{
  if (receivers.len() == 0)
  {
    return
  }
  when (r = receivers(0))
  {
    assert(r.rc() < 10_000)
    forward(receivers.trim(1))
  }
}

main()
{
  let receivers = new Array[EmptyCown](3)
  for i in Range(0, 3)
  {
    receivers(i) = EmptyCown.create()
  }
  for i in Range(0, 10)
  {
    let sender = cown.create(new Sender(10_000))
    sender(receivers)
  }
}
*/

#include "../../../verona.h"
#include "backpressure.h"

namespace backpressure::fanin
{
  using namespace verona::rt;

  static std::vector<EmptyCown*> receivers = {};

  struct Sender : public VCown<Sender>
  {
    size_t messages;

    Sender(size_t messages_) : messages(messages_)
    {
      Cown::acquire(receivers[0]);
    }

    ~Sender()
    {
      Cown::release(ThreadAlloc::get_noncachable(), receivers[0]);
    }
  };

  void forward(std::vector<EmptyCown*> receivers)
  {
    if (receivers.empty())
      return;

    auto* r = receivers[0];
    Cown::acquire(r);
    Cown::schedule<TestBehaviour>(r, [=]() mutable {
      // assert(r->debug_rc() < 10'000);
      receivers.erase(receivers.begin());
      forward(receivers);
      Cown::release(ThreadAlloc::get_noncachable(), r);
    });
  }

  void sender_be(Sender* sender)
  {
    auto* alloc = ThreadAlloc::get();
    forward(receivers);
    sender->messages--;
    if (sender->messages > 0)
      Cown::schedule<TestBehaviour>(sender, [=] { sender_be(sender); });
    else
      Cown::release(alloc, sender);
  }

  void test()
  {
    auto* alloc = ThreadAlloc::get();
    const size_t receiver_count = 3;
    for (size_t r = 0; r < receiver_count; r++)
    {
      receivers.push_back(new (alloc) EmptyCown());
      if (r > 0)
        receivers[r - 1]->add_ref(receivers.back());
    }

    for (size_t s = 0; s < 10; s++)
    {
      auto* sender = new (alloc) Sender(10'000);
      Cown::schedule<TestBehaviour>(sender, [=] { sender_be(sender); });
    }

    Cown::release(alloc, receivers.front());
  }
}
