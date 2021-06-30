// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <ctime>
#include <test/harness.h>

struct TimerPoller : public VCown<TimerPoller>
{
  std::time_t start;
  Cown* owner;

  TimerPoller() : start(std::time(nullptr)) {}

  void notified(Object* o)
  {
    TimerPoller* p = (TimerPoller*)o;
    if (std::difftime(std::time(nullptr), p->start) > 1)
    {
      Cown::poller_remove(p->owner, p);
      std::cout << "Poller Remove\n";
    }
  }
};

void timer_poller()
{
  schedule_lambda([]() {
    std::cout << "Poller Add\n";
    auto alloc = ThreadAlloc::get();
    TimerPoller* p = new (alloc) TimerPoller;

    Cown* owner = Cown::poller_add<YesTransfer>(p);
    p->owner = owner;
  });
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(timer_poller);

  return 0;
}
