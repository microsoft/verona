// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include <test/harness.h>

struct A : public VCown<A>
{};

struct B : public VCown<B>
{};

void early_release_first_test()
{
  Cown* cowns[2];
  Systematic::cout() << "Hello early release\n";

  auto* a = new A;
  auto* b = new B;

  cowns[0] = a;
  cowns[1] = b;

  schedule_lambda(2, cowns, [=]() {
    Systematic::cout() << "msg double 1\n";
    a->release_early();
    yield();
    Systematic::cout() << "msg double 2\n";
  });

  schedule_lambda<YesTransfer>(
    a, []() { Systematic::cout() << "msg single\n"; });

  auto& alloc = ThreadAlloc::get();
  Cown::release(alloc, b);
}

void early_release_second_test()
{
  Cown* cowns[2];
  Systematic::cout() << "Hello early release\n";

  auto* a = new A;
  auto* b = new B;

  cowns[0] = a;
  cowns[1] = b;

  schedule_lambda(2, cowns, [=]() {
    Systematic::cout() << "msg double 1\n";
    b->release_early();
    yield();
    Systematic::cout() << "msg double 2\n";
  });

  schedule_lambda<YesTransfer>(
    b, []() { Systematic::cout() << "msg single\n"; });

  auto& alloc = ThreadAlloc::get();
  Cown::release(alloc, a);
}

void early_release_both_test()
{
  Cown* cowns[2];
  Systematic::cout() << "Hello early release\n";

  auto* a = new A;
  auto* b = new B;

  cowns[0] = a;
  cowns[1] = b;

  schedule_lambda(2, cowns, [=]() {
    Systematic::cout() << "msg double 1\n";
    b->release_early();
    yield();
    a->release_early();
    yield();
    Systematic::cout() << "msg double 2\n";
  });

  schedule_lambda<YesTransfer>(
    b, []() { Systematic::cout() << "msg single b\n"; });
  schedule_lambda<YesTransfer>(
    a, []() { Systematic::cout() << "msg single a\n"; });
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(early_release_first_test);
  harness.run(early_release_second_test);
  harness.run(early_release_both_test);

  return 0;
}
