// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>

void promise_test()
{
  auto pp = Promise<int>::create_promise();
  auto rp = std::move(pp.first);
  auto rp2 = rp;
  auto wp = std::move(pp.second);

  schedule_lambda([wp = std::move(wp)]() mutable {
    Promise<int>::fulfill(std::move(wp), 42);
  });

  rp.then([](int val) { Systematic::cout() << val << std::endl; });
  rp.then([](int val) { Systematic::cout() << val << std::endl; });
  rp2.then([](int val) { Systematic::cout() << val << std::endl; });
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(promise_test);

  return 0;
}
