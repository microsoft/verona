// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>

struct TestCown : public VCown<TestCown>
{};

void lambda_no_cown()
{
  scheduleLambda([]() { std::cout << "Hello world!\n"; });
}

void lambda_cown()
{
  TestCown* c = new TestCown;
  scheduleLambda(c, []() { std::cout << "Hello world!\n"; });
  Cown::release(ThreadAlloc::get(), c);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(lambda_no_cown);
  harness.run(lambda_cown);

  return 0;
}
