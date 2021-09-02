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

  rp.then(
    [](int val) { Systematic::cout() << val << std::endl; },
    []() { Systematic::cout() << "Error\n"; });
  rp.then(
    [](int val) { Systematic::cout() << val << std::endl; },
    []() { Systematic::cout() << "Error\n"; });
  rp2.then(
    [](int val) { Systematic::cout() << val << std::endl; },
    []() { Systematic::cout() << "Error\n"; });
}

void promise_no_reader()
{
  auto pp = Promise<int>::create_promise();
  auto wp = std::move(pp.second);

  schedule_lambda([wp = std::move(wp)]() mutable {
    Promise<int>::fulfill(std::move(wp), 42);
  });
}

void promise_no_writer()
{
  auto pp = Promise<int>::create_promise();
  auto rp = std::move(pp.first);

  rp.then(
    [](int val) { Systematic::cout() << val << std::endl; },
    []() { Systematic::cout() << "Error\n"; });
}
int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(promise_test);
  harness.run(promise_no_reader);
  harness.run(promise_no_writer);

  return 0;
}
