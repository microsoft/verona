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
      auto& sched = Scheduler::get();
      sched.poller_remove(p->owner, p);
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

    auto& sched = Scheduler::get();
    Cown* owner = (Cown*)sched.poller_add(p);
    p->owner = owner;

    // Cown::release(ThreadAlloc::get(), p);
  });
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(timer_poller);

  return 0;
}
