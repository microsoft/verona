// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>

void promise_test()
{
  auto pp = Promise<int>::create_promise();
  auto rp = std::move(pp.first);
  auto wp = std::move(pp.second);

  schedule_lambda([wp = move(wp)]() mutable {
    Promise<int>::PromiseW::fulfill(std::move(wp), 42);
  });

  Promise<int>::PromiseR::then(
    std::move(rp), [](int val) { Systematic::cout() << val << std::endl; });
}

void shared_promise_test()
{
  auto pp = Promise<int>::create_promise();
  auto srp = Promise<int>::PromiseSharedR(std::move(pp.first));
  auto srp2 = srp;
  auto wp = std::move(pp.second);

  schedule_lambda([wp = move(wp)]() mutable {
    Promise<int>::PromiseW::fulfill(std::move(wp), 42);
  });

  Promise<int>::PromiseR::then(
    std::move(srp), [](int val) { Systematic::cout() << val << std::endl; });

  Promise<int>::PromiseR::then(
    std::move(srp2), [](int val) { Systematic::cout() << val << std::endl; });
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(promise_test);
  harness.run(shared_promise_test);

  return 0;
}
