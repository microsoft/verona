// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>

void promise_test()
{
  auto pp = Promise<int>::create_promise();
  auto rp = std::move(pp.first);
  auto wp = std::move(pp.second);

  schedule_lambda(
    [wp = move(wp)]() mutable { PromiseW<int>::fulfill(std::move(wp), 42); });

  PromiseR<int>::then(
    std::move(rp), [](int val) { std::cout << val << std::endl; });
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(promise_test);

  return 0;
}
