// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

/**
 * This test detects if early release allows for interleavings
 * There is currently no checks that this actually occurs as it
 * is probabalistic it occurs.
 *
 * Running for a set of seeds will generate things such as:
 * ---------------------------
 * ./func-sys-early_release --seed 2070 --seed_count 3
 *    Harness starting.
 *    ./func-sys-early_release --seed 2070 --seed_count 3
 *    Seed: 2070
 *    Time so far: 0 seconds
 *    Seed: 2071
 *    Time so far: 0 seconds
 *    Seed: 2072
 *    Time so far: 0 seconds
 *    Test Harness Finished!
 *    Seed: 2070
 *    Time so far: 0 seconds
 *    Seed: 2071
 *    Time so far: 0 seconds
 *    Seed: 2072
 *    Interleaving occurred
 *    Time so far: 0 seconds
 *    Test Harness Finished!
 *    Seed: 2070
 *    Interleaving occurred
 *    Time so far: 0 seconds
 *    Seed: 2071
 *    Time so far: 0 seconds
 *    Seed: 2072
 *    Time so far: 0 seconds
 *    Test Harness Finished!
 * ---------------------------
 * This shows that interleaving occured on the second test for 2072, and the
 * third test for seed 2070.  We can then inspect a particular seed with
 * ---------------------------
 * ./func-sys-early_release --seed 2072 | grep Early
 *   a Early release: begin
 *     3      Early release: start
 *     3      Early release: finish
 *   a Early release: begin
 *     3      Early release: start
 *       4     Early release: Interleaving occured!
 *     3      Early release: finish
 *   a Early release: begin
 *     3      Early release: start
 *     3      Early release: finish
 * ---------------------------
 */
#include <test/harness.h>

struct A : public VCown<A>
{};

struct B : public VCown<B>
{};

std::atomic<bool> flag = false;
void start()
{
  Systematic::cout() << "Early release: start" << Systematic::endl;
  flag = false;
}

void finished()
{
  flag = true;
  Systematic::cout() << "Early release: finish" << Systematic::endl;
}

void interleave()
{
  // Print if this occurs, before `finished`.
  if (!flag)
  {
    Systematic::cout() << "Early release: Interleaving occured!"
                       << Systematic::endl;
    // Print normally so that it can be searched for across multiple seeds.
    printf("Interleaving occurred\n");
  }
}

void early_release_test(bool first, bool second)
{
  Cown* cowns[2];
  Systematic::cout() << "Early release: begin" << Systematic::endl;
  auto* a = new A;
  auto* b = new B;

  cowns[0] = a;
  cowns[1] = b;

  schedule_lambda(2, cowns, [=]() {
    start();

    if (first)
      a->release_early();
    if (second)
      b->release_early();
    yield();

    finished();
  });

  auto& alloc = ThreadAlloc::get();
  if (first)
    schedule_lambda<YesTransfer>(a, []() { interleave(); });
  else
    Cown::release(alloc, a);

  if (second)
    schedule_lambda<YesTransfer>(b, []() { interleave(); });
  else
    Cown::release(alloc, b);
}

int main(int argc, char** argv)
{
  SystematicTestHarness harness(argc, argv);

  harness.run(early_release_test, true, false);
  harness.run(early_release_test, false, true);
  harness.run(early_release_test, true, true);

  return 0;
}
