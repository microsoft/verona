// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "test/harness.h"
#include "test/log.h"

static std::atomic<size_t> poller_count = 0;

struct TestPoller : public VCown<TestPoller>
{
  size_t count = 0;

  void notified(Object*)
  {
    if (++count == 5)
    {
      Scheduler::remove_poller(this);
      poller_count.fetch_sub(1, std::memory_order_seq_cst);
      Cown::release(ThreadAlloc::get(), this);
    }
    logger::cout() << "[!] notify on " << this << ", count: " << count
                   << std::endl;
  }
};

void test()
{
  poller_count.store(3, std::memory_order_seq_cst);

  auto* alloc = ThreadAlloc::get();
  std::vector<TestPoller*> pollers;

  for (size_t i = 0; i < poller_count; i++)
  {
    pollers.push_back(new (alloc) TestPoller);
    Scheduler::add_poller(pollers.back());
  }
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);
  harness.run(test);
  return 0;
}
